"""
Regression tests for SmartDiskCache based on GitHub Issues #280 and #1357.

Issue #280:  Adding/editing files should only rebuild affected cache entries.
Issue #1357: Multiple images sharing the same text content should be deduped
             via hash (one hash entry with multiple paths in hash_index).
"""

import json
import os
import time

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
    """Provides configurable dummy data for pipeline testing.

    *data* maps output names to lists of values.  ``get_item`` returns
    ``values[index % len(values)]`` so that any index is valid.
    """

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


class MutableDummyDataModule(PipelineModule, RandomAccessPipelineModule):
    """Like DummyDataModule but allows mutating data between epochs."""

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


def _pt_files(cache_dir: str) -> set[str]:
    """Return the set of .pt filenames in cache_dir."""
    return {f for f in os.listdir(cache_dir) if f.endswith(".pt")}


def _pt_mtimes(cache_dir: str) -> dict[str, float]:
    """Return {filename: mtime} for all .pt files in cache_dir."""
    return {
        f: os.path.getmtime(os.path.join(cache_dir, f))
        for f in os.listdir(cache_dir) if f.endswith(".pt")
    }


def _build_pipeline(
    cache_dir,
    dummy_mod,
    split_names,
    aggregate_names,
    *,
    modeltype="testmodel",
    source_path_in_name="image_path",
):
    """Build MGDS pipeline: DummyDataModule -> SmartDiskCache -> Output."""
    cache_mod = SmartDiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=aggregate_names,
        modeltype=modeltype,
        source_path_in_name=source_path_in_name,
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
    return ds


# ---------------------------------------------------------------------------
# Issue #280: Incremental cache rebuild
# ---------------------------------------------------------------------------

class TestAddOneFile:
    """Adding one file to an existing dataset should only create one new .pt."""

    def test_add_one_file_to_dataset(self, tmp_path):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        n = 3

        # Create initial N files
        paths = []
        for i in range(n):
            p = _create_source_file(src_dir, f"img_{i}.bin", f"original content {i}".encode())
            paths.append(p)
        tensors = _make_tensors(n, seed=10)

        cache_dir = str(tmp_path / "cache")

        dummy = MutableDummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n,
        )
        ds = _build_pipeline(cache_dir, dummy, split_names=["latent"], aggregate_names=[])
        _drain(ds)  # build cache for N files

        pt_before = _pt_files(cache_dir)
        mtimes_before = _pt_mtimes(cache_dir)
        assert len(pt_before) == n

        # Add 1 more file
        new_path = _create_source_file(src_dir, f"img_{n}.bin", f"new content {n}".encode())
        new_tensor = _make_tensors(1, seed=999)[0]

        new_paths = paths + [new_path]
        new_tensors = tensors + [new_tensor]

        # Must build a fresh pipeline with the updated file list
        # (SmartDiskCache re-reads cache.json each epoch)
        dummy2 = MutableDummyDataModule(
            data={"latent": new_tensors, "image_path": new_paths},
            length=n + 1,
        )
        ds2 = _build_pipeline(cache_dir, dummy2, split_names=["latent"], aggregate_names=[])

        # Small sleep to ensure mtime granularity distinguishes old vs new .pt
        time.sleep(0.05)
        _drain(ds2)

        pt_after = _pt_files(cache_dir)
        mtimes_after = _pt_mtimes(cache_dir)

        # Should now have N+1 .pt files
        assert len(pt_after) == n + 1, f"Expected {n + 1} .pt files, got {len(pt_after)}"

        # Exactly 1 new .pt file was created
        new_pts = pt_after - pt_before
        assert len(new_pts) == 1, f"Expected 1 new .pt file, got {len(new_pts)}: {new_pts}"

        # All previously-existing .pt files should be untouched (same mtime)
        for f in pt_before:
            assert mtimes_after[f] == mtimes_before[f], \
                f"Existing .pt file {f} was rewritten when only a new file was added"


class TestEditOneCaption:
    """Editing one text file should only rebuild that one .pt entry."""

    def test_edit_one_caption(self, tmp_path):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        n = 4

        # Create N text files with distinct content
        paths = []
        for i in range(n):
            p = _create_source_file(src_dir, f"caption_{i}.txt", f"caption text for image {i}".encode())
            paths.append(p)
        tensors = _make_tensors(n, seed=20)

        cache_dir = str(tmp_path / "cache")

        dummy = MutableDummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n,
        )
        ds = _build_pipeline(cache_dir, dummy, split_names=["latent"], aggregate_names=[])
        _drain(ds)

        pt_before = _pt_files(cache_dir)
        mtimes_before = _pt_mtimes(cache_dir)
        index_before = _read_cache_json(cache_dir)

        # Edit one file (change content AND mtime)
        edit_idx = 1
        time.sleep(0.05)
        with open(paths[edit_idx], "wb") as f:
            f.write(b"EDITED caption with completely new text")

        # Identify which .pt file corresponds to the edited source
        norm_edited = os.path.normpath(paths[edit_idx])
        old_entry = index_before["entries"][norm_edited]
        old_cache_file = old_entry["cache_file"]
        old_pt_name = f"{old_cache_file}_1.pt"

        # Rebuild with same file list
        dummy2 = MutableDummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n,
        )
        ds2 = _build_pipeline(cache_dir, dummy2, split_names=["latent"], aggregate_names=[])

        time.sleep(0.05)
        _drain(ds2)

        index_after = _read_cache_json(cache_dir)
        mtimes_after = _pt_mtimes(cache_dir)

        # The hash for the edited file should have changed
        new_entry = index_after["entries"][norm_edited]
        assert new_entry["hash"] != old_entry["hash"], \
            "Hash should change after editing file content"

        # All other files should have unchanged hashes and mtimes
        for i in range(n):
            if i == edit_idx:
                continue
            norm_p = os.path.normpath(paths[i])
            assert index_after["entries"][norm_p]["hash"] == index_before["entries"][norm_p]["hash"], \
                f"Hash for untouched file {i} should not change"

            # The .pt file for untouched entries should not have been rewritten
            untouched_cache_file = index_before["entries"][norm_p]["cache_file"]
            untouched_pt = f"{untouched_cache_file}_1.pt"
            if untouched_pt in mtimes_before:
                assert mtimes_after.get(untouched_pt) == mtimes_before[untouched_pt], \
                    f"Untouched .pt file {untouched_pt} was rewritten"


# ---------------------------------------------------------------------------
# Issue #280: Cross-concept deduplication
# ---------------------------------------------------------------------------

class TestMoveBetweenConcepts:
    """Copying a file from one concept to another should dedup via hash."""

    def test_move_file_between_concepts(self, tmp_path):
        src_dir = tmp_path / "sources"

        # Concept 1: file A with some content
        concept1_dir = src_dir / "concept1"
        concept1_dir.mkdir(parents=True)
        content = b"shared image content across concepts"
        path_c1 = _create_source_file(concept1_dir, "img_a.bin", content)

        # Build cache for concept 1
        tensors1 = _make_tensors(1, seed=30)
        cache_dir = str(tmp_path / "cache")

        dummy1 = DummyDataModule(
            data={"latent": tensors1, "image_path": [path_c1]},
            length=1,
        )
        ds1 = _build_pipeline(cache_dir, dummy1, split_names=["latent"], aggregate_names=[])
        _drain(ds1)

        pt_after_c1 = _pt_files(cache_dir)
        assert len(pt_after_c1) == 1

        # Concept 2: same content, different path
        concept2_dir = src_dir / "concept2"
        concept2_dir.mkdir(parents=True)
        path_c2 = _create_source_file(concept2_dir, "img_a_copy.bin", content)

        # Build cache including both concepts
        tensors2 = _make_tensors(2, seed=31)
        dummy2 = DummyDataModule(
            data={"latent": tensors2, "image_path": [path_c1, path_c2]},
            length=2,
        )
        ds2 = _build_pipeline(cache_dir, dummy2, split_names=["latent"], aggregate_names=[])
        _drain(ds2)

        index = _read_cache_json(cache_dir)
        norm_c1 = os.path.normpath(path_c1)
        norm_c2 = os.path.normpath(path_c2)

        # Both entries should exist
        assert norm_c1 in index["entries"]
        assert norm_c2 in index["entries"]

        # Both should share the same cache_file (dedup)
        assert index["entries"][norm_c1]["cache_file"] == index["entries"][norm_c2]["cache_file"], \
            "Same content in different concepts should share a cache file"

        # hash_index should list both paths under the same hash
        file_hash = index["entries"][norm_c1]["hash"]
        assert file_hash == index["entries"][norm_c2]["hash"]
        assert norm_c1 in index["hash_index"][file_hash]
        assert norm_c2 in index["hash_index"][file_hash]

        # Only 1 set of .pt files should exist (dedup means no extra .pt)
        pt_after_both = _pt_files(cache_dir)
        assert len(pt_after_both) == 1, \
            f"Expected 1 .pt file (dedup), got {len(pt_after_both)}"


# ---------------------------------------------------------------------------
# Issue #1357: Single text file shared by multiple images
# ---------------------------------------------------------------------------

class TestSingleTextPerConcept:
    """Multiple images sharing the same text file content -> hash dedup."""

    def test_single_text_file_per_concept(self, tmp_path):
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        n_images = 4

        # All images "share" the same text content (e.g., a concept-level caption).
        # Simulate by creating N files with identical content.
        shared_text = b"This is the shared caption text for all images in this concept."
        paths = []
        for i in range(n_images):
            p = _create_source_file(src_dir, f"caption_{i}.txt", shared_text)
            paths.append(p)

        tensors = _make_tensors(n_images, seed=40)
        cache_dir = str(tmp_path / "cache")

        dummy = DummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n_images,
        )
        ds = _build_pipeline(cache_dir, dummy, split_names=["latent"], aggregate_names=[])
        _drain(ds)

        index = _read_cache_json(cache_dir)

        # All N entries should exist
        for p in paths:
            norm = os.path.normpath(p)
            assert norm in index["entries"], f"Missing entry for {norm}"

        # All should share the same hash (identical content)
        hashes = set()
        cache_files = set()
        for p in paths:
            norm = os.path.normpath(p)
            hashes.add(index["entries"][norm]["hash"])
            cache_files.add(index["entries"][norm]["cache_file"])

        assert len(hashes) == 1, \
            f"All identical files should have the same hash, got {len(hashes)} distinct hashes"
        assert len(cache_files) == 1, \
            f"All identical files should share one cache_file, got {len(cache_files)}"

        # hash_index should have one hash entry with all N paths
        the_hash = hashes.pop()
        assert the_hash in index["hash_index"]
        indexed_paths = index["hash_index"][the_hash]
        for p in paths:
            norm = os.path.normpath(p)
            assert norm in indexed_paths, \
                f"Path {norm} missing from hash_index entry"

        # Only 1 .pt file should exist (dedup)
        pt = _pt_files(cache_dir)
        assert len(pt) == 1, \
            f"Expected 1 .pt file for deduplicated content, got {len(pt)}"

    def test_text_dedup_then_edit_one(self, tmp_path):
        """After dedup, editing one caption should create a second .pt file."""
        src_dir = tmp_path / "sources"
        src_dir.mkdir()
        n_images = 3

        shared_text = b"shared caption content for dedup-then-edit test"
        paths = []
        for i in range(n_images):
            p = _create_source_file(src_dir, f"caption_{i}.txt", shared_text)
            paths.append(p)

        tensors = _make_tensors(n_images, seed=50)
        cache_dir = str(tmp_path / "cache")

        dummy = MutableDummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n_images,
        )
        ds = _build_pipeline(cache_dir, dummy, split_names=["latent"], aggregate_names=[])
        _drain(ds)

        assert len(_pt_files(cache_dir)) == 1, "Should start with 1 .pt (all deduped)"

        # Edit one caption so it diverges
        time.sleep(0.05)
        with open(paths[0], "wb") as f:
            f.write(b"UNIQUE caption that differs from the rest")

        dummy2 = MutableDummyDataModule(
            data={"latent": tensors, "image_path": paths},
            length=n_images,
        )
        ds2 = _build_pipeline(cache_dir, dummy2, split_names=["latent"], aggregate_names=[])
        _drain(ds2)

        index = _read_cache_json(cache_dir)
        norm_edited = os.path.normpath(paths[0])
        norm_other = os.path.normpath(paths[1])

        # Edited file should now have a different hash
        assert index["entries"][norm_edited]["hash"] != index["entries"][norm_other]["hash"]

        # And a different cache_file
        assert index["entries"][norm_edited]["cache_file"] != index["entries"][norm_other]["cache_file"]

        # Should now have 2 .pt files: one for edited, one shared by the rest
        assert len(_pt_files(cache_dir)) == 2, \
            f"Expected 2 .pt files after editing one deduped caption, got {len(_pt_files(cache_dir))}"
