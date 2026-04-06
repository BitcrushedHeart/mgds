"""
Tests for SmartDiskCache sourceless training mode.

Sourceless mode allows training from a pre-built cache without needing the
original source files.  The SmartDiskCache reads cache.json and .pt files
directly, bypassing the normal hash/mtime validation pipeline.
"""

import json
import os

import pytest
import torch

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, CACHE_VERSION
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyDataModule(PipelineModule, RandomAccessPipelineModule):
    """Provides configurable dummy data for pipeline testing."""

    def __init__(self, data: dict[str, list], length: int):
        super().__init__()
        self.data = data
        self._length = length

    def length(self) -> int:
        return self._length

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return list(self.data.keys())

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        return {name: values[index % len(values)] for name, values in self.data.items()}


def _make_tensors(n: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    return [torch.randn(4, 4, generator=g) for _ in range(n)]


def _create_source_file(directory, name, content: bytes) -> str:
    path = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def _read_cache_json(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "cache.json"), "r") as f:
        return json.load(f)


def _drain(ds):
    """Run one epoch and return all batches."""
    ds.start_next_epoch()
    return list(ds)


def _build_normal_pipeline(
    tmp_path,
    paths,
    tensors,
    split_names,
    aggregate_names,
    *,
    modeltype="testmodel",
    extra_data=None,
):
    """Build a normal (non-sourceless) pipeline for populating the cache."""
    cache_dir = str(tmp_path / "cache")

    dummy_data = {
        "latent": tensors,
        "image_path": paths,
    }
    if extra_data:
        dummy_data.update(extra_data)

    dummy_mod = DummyDataModule(data=dummy_data, length=len(paths))

    cache_mod = SmartDiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=aggregate_names,
        modeltype=modeltype,
        source_path_in_name="image_path",
    )

    all_output_names = split_names + aggregate_names
    output_mod = OutputPipelineModule(names=all_output_names)

    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "A", "path": "dummy"}],
        settings={},
        definition=[[dummy_mod], [cache_mod], [output_mod]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    return ds, cache_dir


def _build_sourceless_pipeline(
    cache_dir,
    split_names,
    aggregate_names,
    *,
    modeltype="testmodel",
):
    """Build a sourceless pipeline that reads from an existing cache."""
    cache_mod = SmartDiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=aggregate_names,
        modeltype=modeltype,
        sourceless=True,
    )

    all_output_names = split_names + aggregate_names
    output_mod = OutputPipelineModule(names=all_output_names)

    ds = MGDS(
        device=torch.device("cpu"),
        concepts=[{"name": "A", "path": "dummy"}],
        settings={},
        definition=[[cache_mod], [output_mod]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )
    return ds


# ---------------------------------------------------------------------------
# Sourceless tests
# ---------------------------------------------------------------------------

class TestSourceless:
    def _setup_and_cache(self, tmp_path, n=3, seed=100):
        """Create source files, build cache, return (paths, tensors, cache_dir)."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = []
        for i in range(n):
            p = _create_source_file(src_dir, f"img_{i}.bin", f"sourceless content {i}".encode())
            paths.append(p)
        tensors = _make_tensors(n, seed=seed)

        ds, cache_dir = _build_normal_pipeline(
            tmp_path, paths, tensors,
            split_names=["latent"],
            aggregate_names=[],
        )
        _drain(ds)  # build the cache
        return paths, tensors, cache_dir

    def test_sourceless_basic(self, tmp_path):
        """Sourceless pipeline loads data from .pt files without source files."""
        paths, tensors, cache_dir = self._setup_and_cache(tmp_path, n=3)

        # Build a new sourceless pipeline pointing at the same cache
        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
        )

        batches = _drain(ds_sl)
        assert len(batches) == 3, f"Expected 3 batches, got {len(batches)}"

        # Each batch should contain a 'latent' tensor
        for b in batches:
            assert "latent" in b, "Batch missing 'latent' key"
            assert torch.is_tensor(b["latent"]), "'latent' should be a tensor"

    def test_sourceless_empty_cache_error(self, tmp_path):
        """Sourceless mode with empty cache raises RuntimeError."""
        cache_dir = str(tmp_path / "empty_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Write an empty cache.json
        with open(os.path.join(cache_dir, "cache.json"), "w") as f:
            json.dump({"version": CACHE_VERSION, "entries": {}, "hash_index": {}}, f)

        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
        )

        with pytest.raises(RuntimeError, match="cache is empty"):
            _drain(ds_sl)

    def test_sourceless_nonexistent_cache_error(self, tmp_path):
        """Sourceless mode with no cache.json raises RuntimeError."""
        cache_dir = str(tmp_path / "no_cache")
        os.makedirs(cache_dir, exist_ok=True)
        # No cache.json at all -- _load_cache_index returns empty entries

        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
        )

        with pytest.raises(RuntimeError, match="cache is empty"):
            _drain(ds_sl)

    def test_sourceless_missing_pt_error(self, tmp_path):
        """Sourceless mode with a missing .pt file raises RuntimeError."""
        paths, tensors, cache_dir = self._setup_and_cache(tmp_path, n=2)

        # Delete one .pt file
        pt_files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        assert len(pt_files) >= 1
        victim = os.path.join(cache_dir, pt_files[0])
        os.remove(victim)

        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
        )

        with pytest.raises(RuntimeError, match="missing"):
            _drain(ds_sl)

    def test_sourceless_wrong_modeltype_error(self, tmp_path):
        """Sourceless mode with mismatched modeltype raises RuntimeError."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        paths = [_create_source_file(src_dir, "img_0.bin", b"modeltype test")]
        tensors = _make_tensors(1, seed=200)

        # Build cache with modeltype="test"
        ds, cache_dir = _build_normal_pipeline(
            tmp_path, paths, tensors,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="test",
        )
        _drain(ds)

        # Now try sourceless with a different modeltype
        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
            modeltype="different",
        )

        with pytest.raises(RuntimeError, match="modeltype mismatch"):
            _drain(ds_sl)

    def test_sourceless_data_integrity(self, tmp_path):
        """Verify tensor values from sourceless mode match cached data."""
        paths, tensors, cache_dir = self._setup_and_cache(tmp_path, n=3, seed=300)

        # Read .pt files directly to get ground truth
        index = _read_cache_json(cache_dir)
        expected = {}
        for filepath in sorted(index["entries"].keys()):
            entry = index["entries"][filepath]
            pt_path = os.path.join(cache_dir, f"{entry['cache_file']}_1.pt")
            cached = torch.load(pt_path, weights_only=False, map_location="cpu")
            expected[filepath] = cached["latent"]

        # Load via sourceless pipeline
        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=[],
        )
        batches = _drain(ds_sl)

        # Collect all latent values from batches
        loaded_latents = [b["latent"].cpu() for b in batches]
        expected_latents = list(expected.values())

        assert len(loaded_latents) == len(expected_latents), \
            f"Expected {len(expected_latents)} items, got {len(loaded_latents)}"

        # Each expected tensor should appear in the loaded set
        for exp_t in expected_latents:
            found = any(torch.equal(exp_t, got_t) for got_t in loaded_latents)
            assert found, "A cached tensor was not found in sourceless output"

    def test_sourceless_with_aggregate_names(self, tmp_path):
        """Sourceless mode correctly loads both split and aggregate names."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        n = 2
        paths = [
            _create_source_file(src_dir, f"img_{i}.bin", f"agg content {i}".encode())
            for i in range(n)
        ]
        tensors = _make_tensors(n, seed=400)

        ds, cache_dir = _build_normal_pipeline(
            tmp_path, paths, tensors,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
            extra_data={"crop_resolution": [(64, 64)] * n},
        )
        _drain(ds)

        # Sourceless load
        ds_sl = _build_sourceless_pipeline(
            cache_dir,
            split_names=["latent"],
            aggregate_names=["crop_resolution"],
        )
        batches = _drain(ds_sl)

        assert len(batches) == n
        for b in batches:
            assert "latent" in b
            assert "crop_resolution" in b
