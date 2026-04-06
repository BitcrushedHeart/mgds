import json
import os
import torch
import pytest
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, CACHE_VERSION
from mgds.OutputPipelineModule import OutputPipelineModule
from tests.conftest import FakeSourceModule, build_pipeline, create_test_files, make_latent_fn


class TestGarbageCollection:
    def _build_cache(self, source_dir, cache_dir, files):
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

    def test_no_orphans_when_all_files_exist(self, source_dir, cache_dir):
        files = ["img1.bin", "img2.bin"]
        self._build_cache(source_dir, cache_dir, files)

        stats = SmartDiskCache.gc_preview(cache_dir)
        assert stats['orphan_count'] == 0
        assert stats['orphan_bytes'] == 0

    def test_orphan_detected_when_source_deleted(self, source_dir, cache_dir):
        files = ["img1.bin", "img2.bin"]
        self._build_cache(source_dir, cache_dir, files)

        os.remove(os.path.join(source_dir, "img1.bin"))

        stats = SmartDiskCache.gc_preview(cache_dir)
        assert stats['orphan_count'] >= 1
        assert stats['orphan_bytes'] > 0

    def test_gc_clean_removes_orphans(self, source_dir, cache_dir):
        files = ["img1.bin", "img2.bin"]
        self._build_cache(source_dir, cache_dir, files)

        os.remove(os.path.join(source_dir, "img1.bin"))

        pt_before = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]

        SmartDiskCache.gc_clean(cache_dir)

        pt_after = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
        assert len(pt_after) < len(pt_before)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index = json.load(f)
        fp2 = os.path.normpath(os.path.join(source_dir, "img2.bin"))
        assert len(index['entries']) == 1
        assert fp2 in index['entries']

    def test_gc_preserves_active_entries(self, source_dir, cache_dir):
        files = ["img1.bin", "img2.bin"]
        self._build_cache(source_dir, cache_dir, files)

        os.remove(os.path.join(source_dir, "img1.bin"))

        SmartDiskCache.gc_clean(cache_dir)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index = json.load(f)
        fp2 = os.path.normpath(os.path.join(source_dir, "img2.bin"))
        entry = index['entries'][fp2]
        pt_path = os.path.join(cache_dir, f"{entry['cache_file']}_1.pt")
        assert os.path.isfile(pt_path)

    def test_gc_detects_orphan_pt_without_cache_json_entry(self, source_dir, cache_dir):
        files = ["img1.bin"]
        self._build_cache(source_dir, cache_dir, files)

        rogue_path = os.path.join(cache_dir, "rogue_abcdef_1.pt")
        torch.save({"fake": torch.zeros(1)}, rogue_path)

        stats = SmartDiskCache.gc_preview(cache_dir)
        assert stats['orphan_count'] >= 1

        SmartDiskCache.gc_clean(cache_dir)
        assert not os.path.exists(rogue_path)

    def test_gc_empty_cache_dir(self, cache_dir):
        stats = SmartDiskCache.gc_preview(cache_dir)
        assert stats['orphan_count'] == 0
        assert stats['orphan_bytes'] == 0

    def test_gc_dedup_keeps_shared_pt(self, source_dir, cache_dir):
        """When two files share a .pt (dedup), deleting one source should not delete the .pt."""
        with open(os.path.join(source_dir, "a.bin"), 'wb') as f:
            f.write(b"shared")
        with open(os.path.join(source_dir, "b.bin"), 'wb') as f:
            f.write(b"shared")

        source = FakeSourceModule(
            file_dir=source_dir, files=["a.bin", "b.bin"],
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

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index = json.load(f)
        fp_a = os.path.normpath(os.path.join(source_dir, "a.bin"))
        shared_cache_file = index['entries'][fp_a]['cache_file']

        os.remove(os.path.join(source_dir, "a.bin"))

        SmartDiskCache.gc_clean(cache_dir)

        pt_path = os.path.join(cache_dir, f"{shared_cache_file}_1.pt")
        assert os.path.isfile(pt_path)

        with open(os.path.join(cache_dir, 'cache.json'), 'r') as f:
            index_after = json.load(f)
        fp_b = os.path.normpath(os.path.join(source_dir, "b.bin"))
        assert fp_b in index_after['entries']
        assert fp_a not in index_after['entries']
