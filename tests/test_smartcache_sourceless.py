import json
import os
import torch
import pytest
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache, CACHE_VERSION
from mgds.OutputPipelineModule import OutputPipelineModule
from tests.conftest import FakeSourceModule, build_pipeline, create_test_files, make_latent_fn


class TestSourcelessValidation:
    def _build_cache(self, source_dir, cache_dir, files, modeltype='test'):
        """Build a cache normally (with source files present)."""
        create_test_files(source_dir, files)
        source = FakeSourceModule(
            file_dir=source_dir, files=files,
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
        )
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype=modeltype,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([source, cache, output])
        pipeline.start_next_epoch()
        return pipeline

    def test_sourceless_loads_from_cache(self, source_dir, cache_dir):
        files = ["img1.bin", "img2.bin"]
        self._build_cache(source_dir, cache_dir, files)

        # Delete source files
        for f in files:
            os.remove(os.path.join(source_dir, f))

        cache_sl = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache_sl, output])
        pipeline.start_next_epoch()

        item = cache_sl.get_item(0)
        assert 'latent_image' in item
        assert isinstance(item['latent_image'], torch.Tensor)

    def test_sourceless_empty_cache_error(self, cache_dir):
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache, output])
        with pytest.raises(RuntimeError, match="cache is empty"):
            pipeline.start_next_epoch()

    def test_sourceless_old_cache_version_error(self, source_dir, cache_dir):
        files = ["img1.bin"]
        self._build_cache(source_dir, cache_dir, files)

        # Tamper cache_version to 0
        cache_path = os.path.join(cache_dir, 'cache.json')
        with open(cache_path, 'r') as f:
            index = json.load(f)
        for entry in index['entries'].values():
            entry['cache_version'] = 0
        with open(cache_path, 'w') as f:
            json.dump(index, f)

        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache, output])
        with pytest.raises(RuntimeError, match="older format"):
            pipeline.start_next_epoch()

    def test_sourceless_wrong_modeltype_error(self, source_dir, cache_dir):
        files = ["img1.bin"]
        self._build_cache(source_dir, cache_dir, files, modeltype='SDXL')

        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='ZImage', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache, output])
        with pytest.raises(RuntimeError, match="modeltype mismatch"):
            pipeline.start_next_epoch()

    def test_sourceless_missing_pt_file_error(self, source_dir, cache_dir):
        """If variation 0 .pt file is missing, sourceless startup must error clearly."""
        files = ["img1.bin"]
        self._build_cache(source_dir, cache_dir, files)

        # Delete the .pt file
        for f in os.listdir(cache_dir):
            if f.endswith('.pt'):
                os.remove(os.path.join(cache_dir, f))

        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache, output])
        with pytest.raises(RuntimeError, match="missing"):
            pipeline.start_next_epoch()
