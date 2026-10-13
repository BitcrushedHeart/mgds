"""Caption-privacy knobs: store_sourceless_prompts=False and scrub_split_names."""

import json
import os
import shutil

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

SECRET = "caption-secret"
PROMPT_NAMES = ("prompt", "prompt_1", "prompt_2")


class CaptionSource(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, image_paths: list[str], text_paths: list[str], captions: list[str]):
        super().__init__()
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.captions = captions

    def length(self) -> int:
        return len(self.image_paths)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [
            "image_path",
            "sample_prompt_path",
            "latent_image",
            "tokens",
            "text_embedding",
            "prompt",
            "prompt_1",
            "prompt_2",
            "concept",
        ]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        caption = self.captions[index]
        return {
            "image_path": self.image_paths[index],
            "sample_prompt_path": self.text_paths[index],
            "latent_image": torch.tensor([index, variation], dtype=torch.float32),
            # Stand-in token ids: distinctive non-zero values per caption.
            "tokens": torch.tensor([1000 + len(caption), 2000 + variation, 7], dtype=torch.int64),
            "text_embedding": torch.tensor([float(len(caption)), variation], dtype=torch.float32),
            "prompt": f"{caption}-v{variation}",
            "prompt_1": f"{caption}-clip-l",
            "prompt_2": f"{caption}-clip-g",
            "concept": {
                "name": "concept-a",
                "path": "concept-a",
                "seed": 1,
                "enabled": True,
                "image_variations": 1,
                "text_variations": 2,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "loss_weight": 1.0,
                "include_subdirectories": False,
                "image": {},
                "text": {},
            },
        }


def _write(path, data: bytes) -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _sources(tmp_path, names_and_captions):
    image_paths, text_paths, captions = [], [], []
    for name, caption in names_and_captions:
        image_paths.append(_write(tmp_path / "src" / f"{name}.png", f"image {name}".encode()))
        text_paths.append(_write(tmp_path / "src" / f"{name}.txt", f"{caption}|{name}".encode()))
        captions.append(caption)
    return image_paths, text_paths, captions


def _dataset(cache_root, sources=None, *, sourceless=False, store_prompts=True, scrub=None):
    common = {
        "balancing_in_name": "concept.balancing",
        "balancing_strategy_in_name": "concept.balancing_strategy",
        "group_enabled_in_name": "concept.enabled",
        "modeltype": "testmodel",
        "sourceless": sourceless,
    }
    image_cache = SmartDiskCache(
        cache_dir=str(cache_root / "image"),
        split_names=["latent_image"],
        aggregate_names=["image_path"],
        variations_in_name="concept.image_variations",
        variations_group_in_name=["concept.path", "concept.seed", "concept.image"],
        source_path_in_name="image_path",
        **common,
    )
    text_cache = SmartDiskCache(
        cache_dir=str(cache_root / "text"),
        split_names=["tokens", "text_embedding"],
        aggregate_names=[],
        variations_in_name="concept.text_variations",
        variations_group_in_name=["concept.path", "concept.seed", "concept.text"],
        source_path_in_name="sample_prompt_path",
        content_key_in_name="prompt",
        store_sourceless_prompts=store_prompts,
        scrub_split_names=scrub,
        **common,
    )
    first = PlaceholderModule() if sourceless else CaptionSource(*sources)
    names = ["image_path", "latent_image", "tokens", "text_embedding", *PROMPT_NAMES]
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "concept-a", "path": "concept-a"}],
        settings={},
        definition=[[first, image_cache, text_cache], OutputPipelineModule(names=names)],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _drain(ds):
    ds.start_next_epoch()
    return list(ds)


def _text_dir(cache_root):
    return cache_root / "text"


def _text_pts(cache_root):
    return sorted(p for p in _text_dir(cache_root).glob("*.pt") if p.name != "blank_sentinel.pt")


def _index_prompt_values(cache_root):
    index = json.loads((_text_dir(cache_root) / "cache.json").read_text(encoding="utf-8"))
    return [
        {k: block.get(k) for k in PROMPT_NAMES if k in block}
        for entry in index["entries"].values()
        for row in (entry.get("sourceless_rows") or {}).values()
        for block in (row.get("runtime_values") or {}).values()
    ]


def _no_secret_bytes(directory) -> bool:
    return not any(path.is_file() and SECRET.encode() in path.read_bytes() for path in directory.iterdir())


def test_scrubbed_build_stores_no_caption_text(tmp_path):
    sources = _sources(tmp_path, [("a", f"{SECRET} alpha"), ("b", f"{SECRET} beta words")])
    cache_root = tmp_path / "cache"
    _drain(_dataset(cache_root, sources, store_prompts=False, scrub=["tokens"]))

    values = _index_prompt_values(cache_root)
    assert values, "rows must still carry runtime_values"
    assert all(v == dict.fromkeys(PROMPT_NAMES, "") for v in values)

    pts = _text_pts(cache_root)
    assert len(pts) == 4  # 2 files x 2 text variations
    for pt in pts:
        data = torch.load(pt, weights_only=False, map_location="cpu")
        assert data["tokens"].shape == (3,) and data["tokens"].dtype == torch.int64
        assert torch.count_nonzero(data["tokens"]) == 0
        assert data["text_embedding"][0] > 0  # unscrubbed names keep their values
        assert all(data["__sourceless_values"][name] == "" for name in PROMPT_NAMES)

    assert _no_secret_bytes(_text_dir(cache_root))


def test_sourceless_serves_blank_prompts_and_zero_tokens(tmp_path):
    sources = _sources(tmp_path, [("a", f"{SECRET} alpha"), ("b", f"{SECRET} beta words")])
    cache_root = tmp_path / "cache"
    builder = _dataset(cache_root, sources, store_prompts=False, scrub=["tokens"])
    _drain(builder)
    sourced = _drain(_dataset(cache_root, sources, store_prompts=False, scrub=["tokens"]))

    shutil.rmtree(tmp_path / "src")
    sourceless = _drain(_dataset(cache_root, sourceless=True, store_prompts=False, scrub=["tokens"]))

    assert len(sourceless) == len(sourced) == 2
    for got, expected in zip(sourceless, sourced, strict=True):
        assert got["image_path"] == expected["image_path"]
        assert torch.equal(got["latent_image"], expected["latent_image"])
        assert torch.equal(got["text_embedding"], expected["text_embedding"])
        assert got["tokens"].shape == expected["tokens"].shape
        assert torch.count_nonzero(got["tokens"]) == 0
        assert all(got[name] == "" for name in PROMPT_NAMES)


def test_enabling_knobs_keeps_entries_and_blanks_existing_index(tmp_path):
    sources = _sources(tmp_path, [("a", f"{SECRET} alpha"), ("b", f"{SECRET} beta words")])
    cache_root = tmp_path / "cache"
    _drain(_dataset(cache_root, sources))  # default: prompts stored, tokens real
    assert any(v["prompt"] for v in _index_prompt_values(cache_root))
    before = {p.name: p.stat().st_mtime_ns for p in _text_pts(cache_root)}

    _drain(_dataset(cache_root, sources, store_prompts=False, scrub=["tokens"]))

    # No rebuild: flipping the knobs is not a cache-invalidating change.
    assert {p.name: p.stat().st_mtime_ns for p in _text_pts(cache_root)} == before
    # The index is blanked on load and saved without a prompt-bearing .bak.
    assert all(v == dict.fromkeys(PROMPT_NAMES, "") for v in _index_prompt_values(cache_root))
    index_bytes = (_text_dir(cache_root) / "cache.json").read_bytes()
    assert SECRET.encode() not in index_bytes
    bak = _text_dir(cache_root) / "cache.json.bak"
    assert not bak.exists() or SECRET.encode() not in bak.read_bytes()

    # Sourceless still loads the (unrewritten) entries.
    shutil.rmtree(tmp_path / "src")
    rows = _drain(_dataset(cache_root, sourceless=True, store_prompts=False, scrub=["tokens"]))
    assert len(rows) == 2
    assert all(row["prompt"] == "" for row in rows)


def test_content_reuse_copy_from_old_donor_is_scrubbed(tmp_path):
    shared = f"{SECRET} shared"
    sources = _sources(tmp_path, [("a", shared)])
    cache_root = tmp_path / "cache"
    _drain(_dataset(cache_root, sources))  # donor written before the knobs
    donor_names = {p.name for p in _text_pts(cache_root)}

    sources = _sources(tmp_path, [("a", shared), ("b", shared)])
    _drain(_dataset(cache_root, sources, store_prompts=False, scrub=["tokens"]))

    new_pts = [p for p in _text_pts(cache_root) if p.name not in donor_names]
    assert len(new_pts) == 2
    for pt in new_pts:
        data = torch.load(pt, weights_only=False, map_location="cpu")
        assert torch.count_nonzero(data["tokens"]) == 0
        assert all(data["__sourceless_values"][name] == "" for name in PROMPT_NAMES)
