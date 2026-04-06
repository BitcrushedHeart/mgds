import json
import os
import torch
import torch.nn.functional as F
import pytest
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
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


class TestSourcelessConceptMetadata:
    def _build_concept_cache(self, source_dir, cache_dir, files, concept_dict):
        create_test_files(source_dir, files)
        concepts_per_file = [concept_dict] * len(files)
        source = FakeSourceModule(
            file_dir=source_dir, files=files,
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
            concepts_per_file=concepts_per_file,
        )
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output = OutputPipelineModule(names=[
            'latent_image',
            ('concept.loss_weight', 'loss_weight'),
            ('concept.type', 'concept_type'),
        ])
        pipeline = build_pipeline([source, cache, output])
        pipeline.start_next_epoch()
        return pipeline

    def test_sourceless_concept_metadata_available(self, source_dir, cache_dir):
        concept = {'loss_weight': 1.5, 'type': 'STANDARD', 'name': 'test', 'path': source_dir, 'seed': 42}
        files = ["img1.bin", "img2.bin"]
        self._build_concept_cache(source_dir, cache_dir, files, concept)

        for f in files:
            os.remove(os.path.join(source_dir, f))

        cache_sl = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=[
            'latent_image',
            ('concept.loss_weight', 'loss_weight'),
            ('concept.type', 'concept_type'),
        ])
        pipeline = build_pipeline([cache_sl, output])
        pipeline.start_next_epoch()

        item = cache_sl.get_item(0)
        assert 'concept' in item
        assert item['concept']['loss_weight'] == 1.5
        assert item['concept']['type'] == 'STANDARD'

    def test_sourceless_concept_values_match_original(self, tmp_path):
        source_dir = str(tmp_path / "source")
        cache_dir = str(tmp_path / "cache")
        os.makedirs(source_dir)
        os.makedirs(cache_dir)

        concept_a = {'loss_weight': 1.0, 'type': 'STANDARD', 'name': 'a', 'path': '', 'seed': 1}
        concept_b = {'loss_weight': 2.5, 'type': 'STANDARD', 'name': 'b', 'path': '', 'seed': 2}

        files = ["img_a1.bin", "img_a2.bin", "img_b1.bin"]
        create_test_files(source_dir, files)
        concepts_per_file = [concept_a, concept_a, concept_b]

        source = FakeSourceModule(
            file_dir=source_dir, files=files,
            split_data={'latent_image': make_latent_fn()},
            source_path_out_name='image_path',
            concepts_per_file=concepts_per_file,
        )
        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            source_path_in_name='image_path', modeltype='test',
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([source, cache, output])
        pipeline.start_next_epoch()

        for f in files:
            os.remove(os.path.join(source_dir, f))

        cache_sl = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output2 = OutputPipelineModule(names=['latent_image'])
        pipeline2 = build_pipeline([cache_sl, output2])
        pipeline2.start_next_epoch()

        loss_weights = set()
        for i in range(3):
            item = cache_sl.get_item(i)
            assert 'concept' in item
            loss_weights.add(item['concept']['loss_weight'])
        assert 1.0 in loss_weights
        assert 2.5 in loss_weights

    def test_sourceless_mock_training_step(self, source_dir, cache_dir):
        concept = {'loss_weight': 1.0, 'type': 'STANDARD', 'name': 'test', 'path': '', 'seed': 42}
        files = ["img1.bin"]
        self._build_concept_cache(source_dir, cache_dir, files, concept)

        for f in files:
            os.remove(os.path.join(source_dir, f))

        cache_sl = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=[
            'latent_image',
            ('concept.loss_weight', 'loss_weight'),
            ('concept.type', 'concept_type'),
        ])
        pipeline = build_pipeline([cache_sl, output])
        pipeline.start_next_epoch()

        batch = next(iter(pipeline))
        assert 'latent_image' in batch
        assert 'loss_weight' in batch

        latent = batch['latent_image'].requires_grad_(True)
        loss_weight = batch['loss_weight']
        target = torch.zeros_like(latent)
        loss = F.mse_loss(latent, target) * loss_weight
        loss.backward()
        assert latent.grad is not None

    def test_sourceless_old_cache_version_rejected(self, source_dir, cache_dir):
        concept = {'loss_weight': 1.0, 'type': 'STANDARD', 'name': 'test', 'path': '', 'seed': 42}
        files = ["img1.bin"]
        self._build_concept_cache(source_dir, cache_dir, files, concept)

        cache_path = os.path.join(cache_dir, 'cache.json')
        with open(cache_path, 'r') as f:
            index = json.load(f)
        for entry in index['entries'].values():
            entry['cache_version'] = 1
        with open(cache_path, 'w') as f:
            json.dump(index, f)

        cache = SmartDiskCache(
            cache_dir=cache_dir, split_names=['latent_image'],
            modeltype='test', sourceless=True,
        )
        output = OutputPipelineModule(names=['latent_image'])
        pipeline = build_pipeline([cache, output])
        with pytest.raises(RuntimeError, match="older format|Rebuild"):
            pipeline.start_next_epoch()
