"""Quick smoke test for SmartDiskCache as DiskCache replacement."""
import json
import os
import tempfile

import torch

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class DummyData(PipelineModule, RandomAccessPipelineModule):
    def __init__(self):
        super().__init__()

    def length(self):
        return 4

    def get_inputs(self):
        return []

    def get_outputs(self):
        return ['latent', 'crop_resolution', 'image_path']

    def get_item(self, variation, index, requested_name=None):
        g = torch.Generator()
        g.manual_seed(index * 100 + variation)
        return {
            'latent': torch.randn(4, 4, generator=g),
            'crop_resolution': (64, 64),
            'image_path': f'fake/img_{index}.png',
        }


def test_smartcache_basic(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    ds = MGDS(
        device=torch.device('cpu'),
        concepts=[{'name': 'test', 'path': 'dummy'}],
        settings={},
        definition=[[DummyData()], [SmartDiskCache(
            cache_dir=cache_dir,
            split_names=['latent'],
            aggregate_names=['crop_resolution', 'image_path'],
            modeltype='test',
            source_path_in_name='image_path',
        )], [OutputPipelineModule(names=['latent', 'crop_resolution'])]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )

    ds.start_next_epoch()
    batches = list(ds)
    assert len(batches) == 4

    ds.start_next_epoch()
    batches2 = list(ds)
    assert len(batches2) == 4


def test_smartcache_creates_cache_json(tmp_path):
    cache_dir = str(tmp_path / 'cache')
    ds = MGDS(
        device=torch.device('cpu'),
        concepts=[{'name': 'test', 'path': 'dummy'}],
        settings={},
        definition=[[DummyData()], [SmartDiskCache(
            cache_dir=cache_dir,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            modeltype='test',
            source_path_in_name='image_path',
        )], [OutputPipelineModule(names=['latent', 'crop_resolution'])]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )

    ds.start_next_epoch()
    list(ds)

    cache_json = os.path.join(cache_dir, 'cache.json')
    assert os.path.isfile(cache_json), "cache.json should be created"

    with open(cache_json) as f:
        index = json.load(f)

    assert 'entries' in index
    assert 'hash_index' in index
    assert index.get('version') == 1


def test_smartcache_no_source_path_fallback(tmp_path):
    """When source files don't exist, SmartDiskCache should fall back to
    pulling data directly from previous modules."""
    cache_dir = str(tmp_path / 'cache')
    ds = MGDS(
        device=torch.device('cpu'),
        concepts=[{'name': 'test', 'path': 'dummy'}],
        settings={},
        definition=[[DummyData()], [SmartDiskCache(
            cache_dir=cache_dir,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            modeltype='test',
            source_path_in_name='image_path',
        )], [OutputPipelineModule(names=['latent', 'crop_resolution'])]],
        batch_size=1,
        state=PipelineState(),
        seed=42,
    )

    ds.start_next_epoch()
    batches = list(ds)
    assert len(batches) == 4
    for batch in batches:
        assert 'latent' in batch
        assert 'crop_resolution' in batch
