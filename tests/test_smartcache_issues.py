import json
import os
import shutil
import time
import torch
import pytest
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, CACHE_VERSION
from mgds.OutputPipelineModule import OutputPipelineModule
from tests.conftest import FakeSourceModule, build_pipeline, create_test_files, make_latent_fn


class TestIssue280:
    """Issue #280: Adding one image to a large dataset should not recache everything."""

    def _run_epoch(self, source_dir, cache_dir, files):
        source = FakeSourceModule(
            file_dir=source_dir, files=files,
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
        )
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([source, cache, output])
        pipeline.start_next_epoch()
        return cache

    def test_add_one_file_to_100_file_dataset(self, source_dir, cache_dir):
        n = 100
        files = [f"img_{i:04d}.bin" for i in range(n)]
        create_test_files(source_dir, files, content_fn=lambda f: f.encode('utf-8'))
        self._run_epoch(source_dir, cache_dir, files)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index1 = json.load(f)
        assert len(index1['entries']) == n

        # Add one new file
        new_file = "img_0100.bin"
        create_test_files(source_dir, [new_file], content_fn=lambda f: b"brand new image")
        files2 = files + [new_file]
        self._run_epoch(source_dir, cache_dir, files2)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index2 = json.load(f)
        assert len(index2['entries']) == n + 1

        # All original entries should be unchanged (same hash)
        for f in files:
            fp = os.path.normpath(os.path.join(source_dir, f))
            assert index1['entries'][fp]['hash'] == index2['entries'][fp]['hash']

    def test_edit_one_caption(self, source_dir, cache_dir):
        files = ["caption_1.txt", "caption_2.txt", "caption_3.txt"]
        create_test_files(source_dir, files)
        self._run_epoch(source_dir, cache_dir, files)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index1 = json.load(f)

        # Edit one caption
        time.sleep(0.05)
        with open(os.path.join(source_dir, "caption_2.txt"), 'wb') as f:
            f.write(b"edited caption text")

        self._run_epoch(source_dir, cache_dir, files)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index2 = json.load(f)

        fp1 = os.path.normpath(os.path.join(source_dir, "caption_1.txt"))
        fp2 = os.path.normpath(os.path.join(source_dir, "caption_2.txt"))
        fp3 = os.path.normpath(os.path.join(source_dir, "caption_3.txt"))

        assert index1['entries'][fp1]['hash'] == index2['entries'][fp1]['hash']
        assert index1['entries'][fp2]['hash'] != index2['entries'][fp2]['hash']
        assert index1['entries'][fp3]['hash'] == index2['entries'][fp3]['hash']


class TestIssue1357:
    """Issue #1357: Moving files between concepts (same content) should reuse cache."""

    def test_move_file_reuses_cache(self, tmp_path):
        concept_a = str(tmp_path / "concept_a")
        concept_b = str(tmp_path / "concept_b")
        cache_dir = str(tmp_path / "cache")
        os.makedirs(concept_a)
        os.makedirs(concept_b)
        os.makedirs(cache_dir)

        with open(os.path.join(concept_a, "img.bin"), 'wb') as f:
            f.write(b"image data")

        source_a = FakeSourceModule(
            file_dir=concept_a, files=["img.bin"],
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
        )
        cache_a = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([source_a, cache_a, output])
        pipeline.start_next_epoch()

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index1 = json.load(f)
        fp_a = os.path.normpath(os.path.join(concept_a, "img.bin"))
        original_cache_file = index1['entries'][fp_a]['cache_file']

        # Move file to concept_b (same content, different path)
        shutil.copy2(os.path.join(concept_a, "img.bin"), os.path.join(concept_b, "img.bin"))

        source_b = FakeSourceModule(
            file_dir=concept_b, files=["img.bin"],
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
        )
        cache_b = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output2 = OutputPipelineModule(names=['latent_image'])
        pipeline2 = build_pipeline([source_b, cache_b, output2])
        pipeline2.start_next_epoch()

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index2 = json.load(f)
        fp_b = os.path.normpath(os.path.join(concept_b, "img.bin"))
        assert fp_b in index2['entries']
        assert index2['entries'][fp_b]['cache_file'] == original_cache_file

    def test_config_change_no_recache(self, source_dir, cache_dir):
        """Switching non-cache config (LR, epochs) should not trigger recaching."""
        files = ["img.bin"]
        create_test_files(source_dir, files)

        source = FakeSourceModule(
            file_dir=source_dir, files=files,
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
        )
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([source, cache, output])
        pipeline.start_next_epoch()

        pt_files_before = set(f for f in os.listdir(cache_dir) if f.endswith('.pt'))
        mtimes_before = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files_before}

        # Re-run with same source files (simulating LR/epoch change)
        cache2 = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output2 = OutputPipelineModule(names=['latent_image'])
        pipeline2 = build_pipeline([source, cache2, output2])
        pipeline2.start_next_epoch()

        pt_files_after = set(f for f in os.listdir(cache_dir) if f.endswith('.pt'))
        mtimes_after = {f: os.path.getmtime(os.path.join(cache_dir, f)) for f in pt_files_after}

        assert pt_files_before == pt_files_after
        for f in pt_files_before:
            assert mtimes_before[f] == mtimes_after[f]
