import json
import os

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.ConceptCacheRouter import ConceptCacheRouter, _IndexMappedAspectBucketing
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModules.VariationSorting import VariationSorting
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest

CONCEPT_COUNT = 3


class ConceptSource(PipelineModule, RandomAccessPipelineModule):
    """Rows interleave concepts (row i belongs to concept i % 3), so each
    partition's rows are non-contiguous in the pipeline's numbering."""

    def __init__(self, image_paths, text_paths, concept_overrides):
        super().__init__()
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.concept_overrides = concept_overrides

    def length(self) -> int:
        return len(self.image_paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [
            "image_path",
            "sample_prompt_path",
            "latent_image",
            "crop_resolution",
            "text_embedding",
            "prompt",
            "concept",
        ]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        concept_index = index % CONCEPT_COUNT
        concept = {
            "name": f"concept-{concept_index}",
            "path": f"concept-{concept_index}",
            "seed": 100 + concept_index,
            "enabled": True,
            "image_variations": 2,
            "text_variations": 2,
            "balancing": 1.0 + concept_index,
            "balancing_strategy": "REPEATS",
            "include_subdirectories": False,
            "image": {},
            "text": {},
            "cache_dir": "",
        }
        concept.update(self.concept_overrides.get(concept_index, {}))
        return {
            "image_path": self.image_paths[index],
            "sample_prompt_path": self.text_paths[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            "crop_resolution": [(512, 768), (768, 512)][(index + variation) % 2],
            "text_embedding": torch.tensor([index + 100, variation], dtype=torch.float32),
            "prompt": f"prompt-{index}-{variation}",
            "concept": concept,
        }


def _image_kwargs(sourceless: bool) -> dict:
    return {
        "split_names": ["latent_image"],
        "aggregate_names": ["crop_resolution", "image_path"],
        "variations_in_name": "concept.image_variations",
        "balancing_in_name": "concept.balancing",
        "balancing_strategy_in_name": "concept.balancing_strategy",
        "variations_group_in_name": ["concept.path", "concept.seed", "concept.image"],
        "group_enabled_in_name": "concept.enabled",
        "modeltype": "testmodel",
        "source_path_in_name": "image_path",
        "sourceless": sourceless,
    }


def _text_kwargs(sourceless: bool) -> dict:
    return {
        "split_names": ["text_embedding"],
        "aggregate_names": [],
        "variations_in_name": "concept.text_variations",
        "balancing_in_name": "concept.balancing",
        "balancing_strategy_in_name": "concept.balancing_strategy",
        "variations_group_in_name": ["concept.path", "concept.seed", "concept.text"],
        "group_enabled_in_name": "concept.enabled",
        "modeltype": "testmodel",
        "source_path_in_name": "sample_prompt_path",
        "sourceless": sourceless,
    }


def _write(path) -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(os.path.basename(str(path)).encode())
    return str(path)


def _source_files(tmp_path, n=9):
    images = [_write(tmp_path / "data" / f"{i:02d}.png") for i in range(n)]
    texts = [_write(tmp_path / "data" / f"{i:02d}.txt") for i in range(n)]
    return images, texts


def _build(
    default_root,
    images=None,
    texts=None,
    overrides=None,
    *,
    routed: bool,
    sourceless: bool = False,
    sourceless_roots=None,
    batch_sorting: bool = False,
):
    if routed:
        caches = [
            ConceptCacheRouter(
                str(default_root), "image", _image_kwargs(sourceless), sourceless_roots=sourceless_roots
            ),
            ConceptCacheRouter(str(default_root), "text", _text_kwargs(sourceless), sourceless_roots=sourceless_roots),
        ]
    else:
        caches = [
            SmartDiskCache(cache_dir=str(default_root / "image"), **_image_kwargs(sourceless)),
            SmartDiskCache(cache_dir=str(default_root / "text"), **_text_kwargs(sourceless)),
        ]

    modules = [PlaceholderModule() if sourceless else ConceptSource(images, texts, overrides or {})]
    modules.extend(caches)
    names = ["image_path", "latent_image", "crop_resolution", "text_embedding"]
    if not sourceless:
        # The uncached names come through VariationSorting, which lays groups
        # out by first row; the routers have to agree with it.
        modules.append(
            VariationSorting(
                names=["prompt", "concept"],
                balancing_in_name="concept.balancing",
                balancing_strategy_in_name="concept.balancing_strategy",
                variations_group_in_name=["concept.path", "concept.seed", "concept.text"],
                group_enabled_in_name="concept.enabled",
            )
        )
        names += ["prompt", "concept"]
    if batch_sorting:
        modules.append(AspectBatchSorting(resolution_in_name="crop_resolution", names=names, batch_size=1))
    return MGDS(
        device=torch.device("cpu"),
        concepts=[],
        settings={},
        definition=[modules, OutputPipelineModule(names=names)],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _epoch(ds):
    ds.start_next_epoch()
    rows = []
    for batch in ds:
        row = (
            batch["image_path"][0] if isinstance(batch["image_path"], list) else batch["image_path"],
            tuple(batch["latent_image"].squeeze(0).tolist()),
            tuple(batch["text_embedding"].squeeze(0).tolist()),
        )
        if "concept" in batch:
            row += (batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"],)
        rows.append(row)
    return rows


def _index_paths(cache_dir) -> set[str]:
    with open(os.path.join(cache_dir, "cache.json"), encoding="utf-8") as f:
        return {os.path.basename(p) for p in json.load(f)["entries"]}


def _overrides(tmp_path):
    return {
        1: {"cache_dir": str(tmp_path / "shared-b")},
        2: {"cache_dir": str(tmp_path / "shared-c")},
    }


@pytest.mark.parametrize("batch_sorting", [False, True])
def test_routed_epochs_match_plain_cache(tmp_path, batch_sorting):
    images, texts = _source_files(tmp_path)
    overrides = _overrides(tmp_path)

    plain = _build(tmp_path / "plain", images, texts, overrides, routed=False, batch_sorting=batch_sorting)
    routed = _build(tmp_path / "run", images, texts, overrides, routed=True, batch_sorting=batch_sorting)

    for _ in range(3):
        plain_rows = _epoch(plain)
        routed_rows = _epoch(routed)
        assert routed_rows == plain_rows
        # Image, text and the VariationSorting names all describe the same row.
        for image_path, latent, text, prompt in routed_rows:
            row = int(latent[0])
            assert os.path.basename(image_path) == f"{row:02d}.png"
            assert int(text[0]) == row + 100
            assert prompt.startswith(f"prompt-{row}-")


def test_each_concept_is_cached_under_its_own_root(tmp_path):
    images, texts = _source_files(tmp_path)
    ds = _build(tmp_path / "run", images, texts, _overrides(tmp_path), routed=True)
    _epoch(ds)

    def rows_of(concept_index, ext):
        return {f"{i:02d}.{ext}" for i in range(len(images)) if i % CONCEPT_COUNT == concept_index}

    assert _index_paths(tmp_path / "run" / "image") == rows_of(0, "png")
    assert _index_paths(tmp_path / "run" / "text") == rows_of(0, "txt")
    assert _index_paths(tmp_path / "shared-b" / "image") == rows_of(1, "png")
    assert _index_paths(tmp_path / "shared-b" / "text") == rows_of(1, "txt")
    assert _index_paths(tmp_path / "shared-c" / "image") == rows_of(2, "png")
    assert _index_paths(tmp_path / "shared-c" / "text") == rows_of(2, "txt")


def test_shared_root_is_reused_by_another_run(tmp_path):
    images, texts = _source_files(tmp_path)
    overrides = _overrides(tmp_path)
    _epoch(_build(tmp_path / "run-1", images, texts, overrides, routed=True))

    shared = tmp_path / "shared-b" / "image"

    def pt_mtimes():
        return {name: os.stat(shared / name).st_mtime_ns for name in os.listdir(shared) if name.endswith(".pt")}

    before = pt_mtimes()
    assert before

    # A second run with a different run cache reuses concept B's latents as-is.
    rows = _epoch(_build(tmp_path / "run-2", images, texts, overrides, routed=True))
    assert rows
    assert pt_mtimes() == before
    assert _index_paths(tmp_path / "run-2" / "image") == {f"{i:02d}.png" for i in range(0, len(images), CONCEPT_COUNT)}


def test_override_spelling_of_default_root_is_one_partition(tmp_path):
    images, texts = _source_files(tmp_path)
    default_root = tmp_path / "run"
    overrides = {1: {"cache_dir": str(default_root) + os.sep}, 2: {"cache_dir": "  "}}
    ds = _build(default_root, images, texts, overrides, routed=True)
    _epoch(ds)

    image_router = next(m for m in ds.loading_pipeline.modules if isinstance(m, ConceptCacheRouter))
    assert len(image_router.children) == 1
    assert _index_paths(default_root / "image") == {os.path.basename(p) for p in images}


def test_disabled_concept_never_touches_its_root(tmp_path):
    images, texts = _source_files(tmp_path)
    overrides = _overrides(tmp_path)
    overrides[2]["enabled"] = False
    rows = _epoch(_build(tmp_path / "run", images, texts, overrides, routed=True))

    assert not os.path.exists(tmp_path / "shared-c")
    assert all(int(latent[0]) % CONCEPT_COUNT != 2 for _, latent, _, _ in rows)


def test_sourceless_serves_every_root(tmp_path):
    images, texts = _source_files(tmp_path)
    overrides = _overrides(tmp_path)
    sourced_rows = _epoch(_build(tmp_path / "run", images, texts, overrides, routed=True))

    roots = [str(tmp_path / "run"), overrides[1]["cache_dir"], overrides[2]["cache_dir"]]
    sourceless = _build(tmp_path / "run", routed=True, sourceless=True, sourceless_roots=roots)
    sourceless_rows = _epoch(sourceless)

    assert sorted(row[:3] for row in sourceless_rows) == sorted(row[:3] for row in sourced_rows)
    for image_path, latent, text in sourceless_rows:
        row = int(latent[0])
        assert os.path.basename(image_path) == f"{row:02d}.png"
        assert int(text[0]) == row + 100


def test_sourceless_only_serves_listed_roots(tmp_path):
    images, texts = _source_files(tmp_path)
    overrides = _overrides(tmp_path)
    _epoch(_build(tmp_path / "run", images, texts, overrides, routed=True))

    rows = _epoch(_build(tmp_path / "run", routed=True, sourceless=True, sourceless_roots=[overrides[1]["cache_dir"]]))
    assert rows
    assert {int(latent[0]) % CONCEPT_COUNT for _, latent, _ in rows} == {1}


def test_aspect_bucketing_sees_pipeline_indices():
    class Recorder:
        bucket_resolutions = {512: []}

        def __init__(self):
            self.calls = []
            self._target_override = None

        def variant_key_from_aspect(self, variation, index, aspect):
            self.calls.append(("aspect", variation, index, aspect))
            return "key"

        def _target_resolutions_for_item(self, variation, index):
            self.calls.append(("targets", variation, index))
            return "512"

        def bucket_for_aspect(self, aspect, target):
            return (target, target)

    inner = Recorder()
    mapped = _IndexMappedAspectBucketing(inner, [4, 7, 10])

    assert mapped.variant_key_from_aspect(3, 1, 0.5) == "key"
    assert mapped._target_resolutions_for_item(0, 2) == "512"
    assert mapped.bucket_for_aspect(1.0, 512) == (512, 512)
    assert mapped.bucket_resolutions == {512: []}
    mapped._target_override = 768
    assert inner._target_override == 768
    assert inner.calls == [("aspect", 3, 7, 0.5), ("targets", 0, 10)]
