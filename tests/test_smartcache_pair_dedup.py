"""Dedup identity of caches with sidecars (DPO pairs, masks).

A cache file is named and deduped by the source bytes PLUS every watched
sidecar (``SmartDiskCache._dedup_key``). Before that, every pair sharing a
chosen image collapsed onto one .pt, so pairs with a different rejected
image or mask trained on another pair's tensors - in sourceless and in normal
(latent_caching) runs alike, since both read the cached .pt. These tests pin
the new rule, the migration of such legacy indexes, and the sourceless
guards.
"""

import json
import os

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import NO_RESOLUTION_KEY, SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch

import pytest
import xxhash


def _content_id(path: str) -> float:
    with open(path, "rb") as f:
        return float(xxhash.xxh64(f.read()).intdigest() % 10007)


class PairSource(PipelineModule, RandomAccessPipelineModule):
    """One DPO pair per index: chosen image, rejected image, optional mask.
    Every latent is a function of the file bytes it stands for, so two pairs
    with identical files produce identical tensors and different files
    different ones."""

    def __init__(self, pairs: list[dict]):
        super().__init__()
        self.pairs = pairs

    def length(self) -> int:
        return len(self.pairs)

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return [
            "image_path",
            "image_path_rejected",
            "mask_path",
            "latent_image",
            "latent_image_rejected",
            "latent_mask",
            "concept",
        ]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        pair = self.pairs[index]
        mask = pair.get("mask_path")
        return {
            "image_path": pair["image_path"],
            "image_path_rejected": pair["rejected_path"],
            "mask_path": mask,
            "latent_image": torch.tensor([_content_id(pair["image_path"]), float(variation)]),
            "latent_image_rejected": torch.tensor([_content_id(pair["rejected_path"]), float(variation)]),
            "latent_mask": torch.tensor([_content_id(mask) if mask else 0.0, float(variation)]),
            "concept": {
                "name": pair["concept"],
                "path": pair["concept"],
                "seed": 7,
                "enabled": True,
                "image_variations": 1,
                "balancing": 1.0,
                "balancing_strategy": "REPEATS",
                "loss_weight": 1.0,
                "include_subdirectories": False,
                "dpo_chosen_pattern": "chosen/{}",
                "dpo_rejected_pattern": "rejected/{}",
                "dpo_beta_override": pair.get("beta", 0.0),
                "image": {"resolution": "test"},
            },
        }


def _write(path, data: bytes) -> str:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return os.path.normpath(str(path))


def _pair(root, concept: str, name: str, chosen: bytes, rejected: bytes, mask: bytes | None = None, beta=0.0):
    pair = {
        "image_path": _write(root / concept / "chosen" / f"{name}.png", chosen),
        "rejected_path": _write(root / concept / "rejected" / f"{name}.png", rejected),
        "concept": concept,
        "beta": beta,
    }
    if mask is not None:
        pair["mask_path"] = _write(root / concept / "chosen" / f"{name}-masklabel.png", mask)
    return pair


def _cache(cache_dir, *, sourceless=False, watched=("image_path_rejected", "mask_path")):
    return SmartDiskCache(
        cache_dir=str(cache_dir),
        split_names=["latent_image", "latent_image_rejected", "latent_mask"],
        aggregate_names=["image_path"],
        variations_in_name="concept.image_variations",
        balancing_in_name="concept.balancing",
        balancing_strategy_in_name="concept.balancing_strategy",
        variations_group_in_name=[
            "concept.path",
            "concept.seed",
            "concept.include_subdirectories",
            "concept.image",
            "concept.dpo_chosen_pattern",
            "concept.dpo_rejected_pattern",
        ],
        group_enabled_in_name="concept.enabled",
        modeltype="testmodel",
        source_path_in_name="image_path",
        extra_watched_paths_in_names=list(watched) or None,
        sourceless=sourceless,
    )


def _dataset(cache_dir, pairs=None, *, sourceless=False, watched=("image_path_rejected", "mask_path")):
    modules = [] if sourceless else [PairSource(pairs)]
    modules.append(_cache(cache_dir, sourceless=sourceless, watched=watched))
    names = ["image_path", "latent_image", "latent_image_rejected", "latent_mask"]
    if sourceless:
        names.append("concept")
    return MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "c", "path": "c"}],
        settings={},
        definition=[modules, OutputPipelineModule(names=names)],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )


def _serve(ds) -> dict[str, dict]:
    ds.start_next_epoch()
    rows = {}
    for item in ds:
        rows[os.path.normpath(item["image_path"])] = item
    return rows


def _index(cache_dir) -> dict:
    with open(os.path.join(cache_dir, "cache.json"), encoding="utf-8") as f:
        return json.load(f)


def _cache_file_of(index: dict, path: str) -> str:
    return index["entries"][path]["variants"][NO_RESOLUTION_KEY]["cache_file"]


def _pt(cache_dir, cache_file: str) -> dict:
    return torch.load(os.path.join(cache_dir, f"{cache_file}_1.pt"), weights_only=False, map_location="cpu")


# --------------------------------------------------------------------------- #
# the key rule
# --------------------------------------------------------------------------- #


def test_one_chosen_image_with_three_rejected_images_gets_three_files(tmp_path):
    root = tmp_path / "data"
    chosen = b"the-same-chosen-image"
    pairs = [
        _pair(root, "OffPolicy", "pair_1", chosen, b"negative-one"),
        _pair(root, "OffPolicy", "pair_2", chosen, b"negative-two"),
        _pair(root, "OffPolicy", "pair_3", chosen, b"negative-three", mask=b"edit-mask"),
    ]
    cache_dir = tmp_path / "cache"
    rows = _serve(_dataset(cache_dir, pairs))

    index = _index(cache_dir)
    files = {_cache_file_of(index, p["image_path"]) for p in pairs}
    assert len(files) == 3, "every pair gets its own file"
    hashes = {index["entries"][p["image_path"]]["hash"] for p in pairs}
    assert len(hashes) == 1  # same chosen bytes ...
    assert len({index["entries"][p["image_path"]]["dedup_key"] for p in pairs}) == 3  # ... different identity
    assert len(index["hash_index"]) == 3
    for pair in pairs:
        own_rejected = _content_id(pair["rejected_path"])
        assert rows[pair["image_path"]]["latent_image_rejected"][0].item() == own_rejected
        stored = _pt(cache_dir, _cache_file_of(index, pair["image_path"]))
        assert stored["latent_image_rejected"][0].item() == own_rejected
        assert os.path.normpath(stored["image_path"]) == pair["image_path"]
    # the masked pair's mask is its own too
    assert rows[pairs[2]["image_path"]]["latent_mask"][0].item() == _content_id(pairs[2]["mask_path"])
    assert rows[pairs[0]["image_path"]]["latent_mask"][0].item() == 0.0


def test_identical_chosen_and_identical_sidecars_still_dedup(tmp_path):
    root = tmp_path / "data"
    pairs = [
        _pair(root, "RLHF", "pair_1", b"chosen", b"negative", mask=b"mask"),
        _pair(root, "Edits", "pair_9", b"chosen", b"negative", mask=b"mask"),
    ]
    cache_dir = tmp_path / "cache"
    rows = _serve(_dataset(cache_dir, pairs))
    index = _index(cache_dir)
    assert _cache_file_of(index, pairs[0]["image_path"]) == _cache_file_of(index, pairs[1]["image_path"])
    assert len([n for n in os.listdir(cache_dir) if n.endswith(".pt") and not n.startswith("blank_sentinel")]) == 1
    ((key, paths),) = index["hash_index"].items()
    assert sorted(paths) == sorted(p["image_path"] for p in pairs)
    assert key == index["entries"][pairs[0]["image_path"]]["dedup_key"]
    for pair in pairs:
        assert rows[pair["image_path"]]["latent_image_rejected"][0].item() == _content_id(pair["rejected_path"])


def test_caches_without_sidecars_keep_the_source_hash_as_key_and_file_name(tmp_path):
    root = tmp_path / "data"
    pairs = [_pair(root, "Plain", "img_1", b"one", b"unused"), _pair(root, "Plain", "img_2", b"two", b"unused")]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs, watched=()))
    index = _index(cache_dir)
    for pair in pairs:
        entry = index["entries"][pair["image_path"]]
        assert entry["dedup_key"] == entry["hash"]
        assert entry["sidecar_hashes"] == {}
        assert _cache_file_of(index, pair["image_path"]) == entry["hash"][:12]
        assert entry["hash"] in index["hash_index"]
    assert SmartDiskCache._dedup_key("abc", None) == "abc"
    assert SmartDiskCache._dedup_key("abc", {}) == "abc"
    assert SmartDiskCache._dedup_key("abc", {"x": "1"}) != "abc"
    assert SmartDiskCache._dedup_key("abc", {"p": "1", "q": "2"}) == SmartDiskCache._dedup_key(
        "abc", {"r": "2", "s": "1"}
    )


# --------------------------------------------------------------------------- #
# legacy indexes: one file shared by pairs of different identity
# --------------------------------------------------------------------------- #


def _forge_legacy_shared_file(cache_dir, pairs: list[dict], holder: dict) -> str:
    """Rewrite a freshly built index into the pre-fix shape: every pair's
    variant points at ``holder``'s file (named by the shared source hash, as
    the old code did), the other pairs' own files are gone, no dedup keys."""
    index = _index(cache_dir)
    holder_entry = index["entries"][holder["image_path"]]
    shared_file = holder_entry["hash"][:12]
    old_holder_file = _cache_file_of(index, holder["image_path"])
    os.replace(os.path.join(cache_dir, f"{old_holder_file}_1.pt"), os.path.join(cache_dir, f"{shared_file}_1.pt"))
    for pair in pairs:
        entry = index["entries"][pair["image_path"]]
        own_file = _cache_file_of(index, pair["image_path"])
        if pair is not holder and own_file != old_holder_file:
            os.remove(os.path.join(cache_dir, f"{own_file}_1.pt"))
        entry["variants"][NO_RESOLUTION_KEY]["cache_file"] = shared_file
        entry.pop("dedup_key", None)
    index["hash_index"] = {holder_entry["hash"]: [p["image_path"] for p in pairs]}
    with open(os.path.join(cache_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return shared_file


def test_legacy_shared_file_serves_the_holder_to_every_twin_until_rebuilt(tmp_path):
    """The non-sourceless (latent_caching) path reads the same .pt, so the
    bug hits it too: under the legacy index every twin loads the holder's
    rejected latent. The migration flags the group, the next normal pass
    rebuilds every non-holder into its own file and leaves the holder's."""
    root = tmp_path / "data"
    chosen = b"shared-chosen"
    pairs = [
        _pair(root, "RLHF_8", "pair_0278", chosen, b"rlhf-negative"),
        _pair(root, "Minor_Edits", "pair_0018", chosen, b"edit-negative", mask=b"edit-mask"),
        _pair(root, "Minor_Edits", "pair_0019", chosen, b"another-edit-negative"),
    ]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs))
    holder = pairs[0]
    shared_file = _forge_legacy_shared_file(cache_dir, pairs, holder)
    holder_pt_mtime = os.path.getmtime(os.path.join(cache_dir, f"{shared_file}_1.pt"))

    # the legacy layout, read the way the old code read it: one file for all
    legacy = _index(cache_dir)
    for pair in pairs:
        assert _cache_file_of(legacy, pair["image_path"]) == shared_file
        stored = _pt(cache_dir, shared_file)
        assert stored["latent_image_rejected"][0].item() == _content_id(holder["rejected_path"])
    assert _content_id(pairs[1]["rejected_path"]) != _content_id(holder["rejected_path"])

    # on load the migration flags the whole group ...
    stamped, flagged = SmartDiskCache._migrate_dedup_keys(legacy)
    assert (stamped, flagged) == (3, 3)
    assert len(legacy["hash_index"]) == 3
    assert all(legacy["entries"][p["image_path"]]["shared_file_conflict"] for p in pairs)

    # ... and a normal cache pass rebuilds the non-holders into their own files
    rows = _serve(_dataset(cache_dir, pairs))
    index = _index(cache_dir)
    assert _cache_file_of(index, holder["image_path"]) == shared_file
    assert os.path.getmtime(os.path.join(cache_dir, f"{shared_file}_1.pt")) == holder_pt_mtime
    files = {_cache_file_of(index, p["image_path"]) for p in pairs}
    assert len(files) == 3
    for pair in pairs:
        assert "shared_file_conflict" not in index["entries"][pair["image_path"]]
        assert rows[pair["image_path"]]["latent_image_rejected"][0].item() == _content_id(pair["rejected_path"])
        stored = _pt(cache_dir, _cache_file_of(index, pair["image_path"]))
        assert os.path.normpath(stored["image_path"]) == pair["image_path"]
    assert rows[pairs[1]["image_path"]]["latent_mask"][0].item() == _content_id(pairs[1]["mask_path"])
    # a second load finds nothing to flag
    assert SmartDiskCache._migrate_dedup_keys(_index(cache_dir)) == (0, 0)


def test_legacy_twins_of_identical_identity_are_not_flagged(tmp_path):
    root = tmp_path / "data"
    pairs = [_pair(root, "A", "p1", b"chosen", b"neg"), _pair(root, "B", "p2", b"chosen", b"neg")]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs))
    index = _index(cache_dir)
    for pair in pairs:
        index["entries"][pair["image_path"]].pop("dedup_key")
    assert SmartDiskCache._migrate_dedup_keys(index) == (2, 0)
    assert not any("shared_file_conflict" in index["entries"][p["image_path"]] for p in pairs)


def test_sourceless_refuses_legacy_shared_files(tmp_path):
    root = tmp_path / "data"
    pairs = [
        _pair(root, "RLHF_8", "pair_1", b"chosen", b"neg-1"),
        _pair(root, "Edits", "pair_2", b"chosen", b"neg-2"),
        _pair(root, "Edits", "pair_3", b"chosen", b"neg-3"),
        _pair(root, "Other", "pair_4", b"different-chosen", b"neg-4"),
    ]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs))
    _forge_legacy_shared_file(cache_dir, pairs[:3], pairs[0])

    with pytest.raises(RuntimeError, match=r"2 pair entries .* share a cache file with a different pair"):
        _serve(_dataset(cache_dir, sourceless=True))

    # the holder and the unrelated pair are fine once the twins are rebuilt
    _serve(_dataset(cache_dir, pairs))
    rows = _serve(_dataset(cache_dir, sourceless=True))
    assert len(rows) == 4
    for pair in pairs:
        assert rows[pair["image_path"]]["latent_image_rejected"][0].item() == _content_id(pair["rejected_path"])


def test_trust_mode_refuses_legacy_shared_files(tmp_path, monkeypatch):
    root = tmp_path / "data"
    pairs = [_pair(root, "A", "p1", b"chosen", b"neg-1"), _pair(root, "B", "p2", b"chosen", b"neg-2")]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs))
    _forge_legacy_shared_file(cache_dir, pairs, pairs[0])
    monkeypatch.setenv("OT_SKIP_CACHE_VALIDATION", "1")
    with pytest.raises(RuntimeError, match=r"Trust mode: 1 pair entries"):
        _serve(_dataset(cache_dir, pairs))


# --------------------------------------------------------------------------- #
# sourceless serving: the entry's own concept, never a twin's
# --------------------------------------------------------------------------- #


def test_sourceless_serving_takes_each_entrys_own_concept(tmp_path):
    root = tmp_path / "data"
    # identical identity -> legitimately one file, built from (and stamped with) the first pair
    pairs = [
        _pair(root, "OnPolicy", "p1", b"chosen", b"neg", beta=40.0),
        _pair(root, "Edits", "p2", b"chosen", b"neg", beta=650.0),
    ]
    cache_dir = tmp_path / "cache"
    _serve(_dataset(cache_dir, pairs))
    index = _index(cache_dir)
    assert _cache_file_of(index, pairs[0]["image_path"]) == _cache_file_of(index, pairs[1]["image_path"])
    stored = _pt(cache_dir, _cache_file_of(index, pairs[0]["image_path"]))
    # whichever pair built the file stamped its own concept into it
    holder = next(p for p in pairs if os.path.normpath(stored["image_path"]) == p["image_path"])
    twin = next(p for p in pairs if p is not holder)
    assert stored["__concept_name"] == holder["concept"]

    rows = _serve(_dataset(cache_dir, sourceless=True))
    for pair in pairs:
        assert rows[pair["image_path"]]["concept"]["name"] == pair["concept"]
        assert rows[pair["image_path"]]["concept"]["dpo_beta_override"] == pair["beta"]

    # With the twin's own row values gone (a cache built under trust mode
    # never stamps them), the .pt's baked concept belongs to the holder and
    # must not be lent out.
    entry = index["entries"][twin["image_path"]]
    for row in entry.get("sourceless_rows", {}).values():
        row.pop("runtime_values", None)
    entry.pop("sourceless_runtime_values", None)
    with open(os.path.join(cache_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    with pytest.raises(RuntimeError, match=r"carries no runtime values of its own and its cache file was built from"):
        _serve(_dataset(cache_dir, sourceless=True))
