"""
Regression tests for mgds DiskCache module.

These tests verify the current behavior of DiskCache before any codebase
changes are made.  They exercise caching, cache reuse, variations, balancing,
aggregate metadata, completeness checks, and multi-concept scenarios.
"""

import hashlib
import json
import math
import os
import time

import pytest
import torch

from mgds.MGDS import MGDS
from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.PipelineModule import PipelineModule, PipelineState
from mgds.pipelineModules.DiskCache import DiskCache
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyDataModule(PipelineModule, RandomAccessPipelineModule):
    """Provides configurable dummy data for pipeline testing.

    *data* maps output names to lists of values.  get_item returns
    ``values[index % len(values)]`` so that any index is valid.

    NOTE: When DiskCache resolves dotted names like ``concept.variations``,
    the PipelineModule infrastructure splits on ``.`` and navigates into the
    returned dict.  So if DiskCache asks for ``concept.variations`` and this
    module outputs ``concept``, the result is ``item['concept']['variations']``.
    Callers should therefore provide a ``'concept'`` key whose values are dicts
    with the appropriate sub-keys (``variations``, ``balancing``, etc.).
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


def _string_key(data: list) -> str:
    """Replicate DiskCache.__string_key for group-hash calculation."""
    json_data = json.dumps(data, sort_keys=True, ensure_ascii=True,
                           separators=(',', ':'), indent=None)
    return hashlib.sha256(json_data.encode('utf-8')).hexdigest()


def _make_tensors(n: int, seed: int = 0) -> list[torch.Tensor]:
    """Return *n* deterministic 4x4 float tensors."""
    g = torch.Generator()
    g.manual_seed(seed)
    return [torch.randn(4, 4, generator=g) for _ in range(n)]


def _build_pipeline(
    tmp_path,
    concepts,
    dummy_data,
    dummy_length,
    split_names,
    aggregate_names,
    *,
    variations_in_name=None,
    balancing_in_name=None,
    balancing_strategy_in_name=None,
    variations_group_in_name=None,
    batch_size=1,
    seed=42,
    settings=None,
):
    """Build an MGDS pipeline with DummyDataModule -> DiskCache -> Output.

    The module order after MGDS flattens is:
      [ConceptPipelineModule, SettingsPipelineModule, DummyDataModule,
       DiskCache, OutputPipelineModule]

    DiskCache resolves names by walking *backwards* from its position.  For
    dotted names like ``concept.variations``, the lookup splits on ``.`` and
    finds the closest module outputting ``concept``.  Because DummyDataModule
    is closer than ConceptPipelineModule, it takes precedence -- which is the
    same pattern as CollectPaths in real pipelines.
    """
    cache_dir = str(tmp_path / 'cache')

    dummy_mod = DummyDataModule(data=dummy_data, length=dummy_length)

    cache_mod = DiskCache(
        cache_dir=cache_dir,
        split_names=split_names,
        aggregate_names=aggregate_names,
        variations_in_name=variations_in_name,
        balancing_in_name=balancing_in_name,
        balancing_strategy_in_name=balancing_strategy_in_name,
        variations_group_in_name=variations_group_in_name,
    )

    all_output_names = split_names + aggregate_names
    output_mod = OutputPipelineModule(names=all_output_names)

    ds = MGDS(
        device=torch.device('cpu'),
        concepts=concepts,
        settings=settings or {},
        definition=[[dummy_mod], [cache_mod], [output_mod]],
        batch_size=batch_size,
        state=PipelineState(),
        seed=seed,
    )
    return ds, cache_dir


def _drain(ds):
    """Run one epoch through *ds* and return all batches as a list."""
    ds.start_next_epoch()
    batches = []
    for batch in ds:
        batches.append(batch)
    return batches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicCaching:
    """Verify that DiskCache creates the expected files on disk."""

    def test_basic_caching(self, tmp_path):
        num_items = 4
        tensors = _make_tensors(num_items)

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'A', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
        )

        _drain(ds)

        # Without variations_in_name, DiskCache uses a single group with key ''
        # and 1 variation => variation-0 directory.
        var_dir = os.path.join(cache_dir, '', 'variation-0')
        assert os.path.isdir(var_dir), f"Expected cache dir {var_dir} to exist"

        # Each item should have a {index}.pt file
        for i in range(num_items):
            pt_file = os.path.join(var_dir, f'{i}.pt')
            assert os.path.isfile(pt_file), f"Missing split cache file {pt_file}"

        # aggregate.pt must exist
        agg_file = os.path.join(var_dir, 'aggregate.pt')
        assert os.path.isfile(agg_file), f"Missing aggregate file {agg_file}"


class TestCacheDataIntegrity:
    """Read back cached .pt files and compare to expected data."""

    def test_cache_data_integrity(self, tmp_path):
        num_items = 3
        tensors = _make_tensors(num_items, seed=123)

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'B', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
        )

        _drain(ds)

        var_dir = os.path.join(cache_dir, '', 'variation-0')

        for i in range(num_items):
            cached = torch.load(
                os.path.join(var_dir, f'{i}.pt'),
                weights_only=False,
                map_location='cpu',
            )
            assert 'latent' in cached
            assert torch.equal(cached['latent'], tensors[i]), \
                f"Tensor mismatch at index {i}"

        # Check aggregate data
        agg = torch.load(
            os.path.join(var_dir, 'aggregate.pt'),
            weights_only=False,
            map_location='cpu',
        )
        assert isinstance(agg, list)
        assert len(agg) == num_items
        for i in range(num_items):
            assert agg[i]['crop_resolution'] == (64, 64)


class TestCacheReuse:
    """Cache should not be recreated on subsequent epochs."""

    def test_cache_reuse(self, tmp_path):
        num_items = 3
        tensors = _make_tensors(num_items)

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'C', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
        )

        # Epoch 1 -- creates cache
        _drain(ds)

        var_dir = os.path.join(cache_dir, '', 'variation-0')

        # Record modification times
        mtimes = {}
        for fname in os.listdir(var_dir):
            fpath = os.path.join(var_dir, fname)
            mtimes[fname] = os.path.getmtime(fpath)

        # Small sleep to ensure mtime would differ if files were rewritten
        time.sleep(0.1)

        # Epoch 2 -- should reuse cache
        _drain(ds)

        for fname in os.listdir(var_dir):
            fpath = os.path.join(var_dir, fname)
            assert os.path.getmtime(fpath) == mtimes[fname], \
                f"File {fname} was modified on second epoch (should be cached)"


class TestCacheWithVariations:
    """When concept.variations > 1, separate variation-N/ dirs should exist."""

    def test_cache_with_variations(self, tmp_path):
        num_items = 4
        num_variations = 2
        tensors = _make_tensors(num_items)

        # DummyDataModule outputs 'concept' as a dict for each item, mimicking
        # how CollectPaths repeats the concept dict for each image file.
        concept_dict = {
            'name': 'VarConcept',
            'variations': num_variations,
            'balancing': 1.0,
            'balancing_strategy': 'REPEATS',
        }

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'VarConcept', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
                'concept': [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            variations_in_name='concept.variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name='concept.name',
        )

        # The group key is SHA256 of the JSON-serialized group values
        group_key = _string_key(['VarConcept'])

        # Drain enough epochs to trigger caching for both variations.
        # With 4 items and 2 variations, epoch 0 (out_variation=0) caches
        # variation-0 and epoch 1 (out_variation=1) caches variation-1.
        _drain(ds)
        _drain(ds)

        for v in range(num_variations):
            var_dir = os.path.join(cache_dir, group_key, f'variation-{v}')
            assert os.path.isdir(var_dir), \
                f"Expected variation dir {var_dir} to exist"
            assert os.path.isfile(os.path.join(var_dir, 'aggregate.pt')), \
                f"Missing aggregate.pt in variation-{v}"
            for i in range(num_items):
                assert os.path.isfile(os.path.join(var_dir, f'{i}.pt')), \
                    f"Missing {i}.pt in variation-{v}"


class TestAggregateData:
    """Verify aggregate.pt contains the expected non-tensor metadata."""

    def test_aggregate_data(self, tmp_path):
        num_items = 3
        tensors = _make_tensors(num_items)
        image_paths = [f'/fake/img_{i}.png' for i in range(num_items)]

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'AggTest', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
                'image_path': image_paths,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution', 'image_path'],
        )

        _drain(ds)

        var_dir = os.path.join(cache_dir, '', 'variation-0')
        agg = torch.load(
            os.path.join(var_dir, 'aggregate.pt'),
            weights_only=False,
            map_location='cpu',
        )

        assert len(agg) == num_items
        for i in range(num_items):
            assert agg[i]['crop_resolution'] == (64, 64)
            assert agg[i]['image_path'] == image_paths[i]


class TestCacheCompletenessCheck:
    """Removing aggregate.pt should trigger cache rebuild."""

    def test_cache_completeness_check(self, tmp_path):
        num_items = 3
        tensors = _make_tensors(num_items)

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'Complete', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
        )

        # Epoch 1 -- build cache
        _drain(ds)

        var_dir = os.path.join(cache_dir, '', 'variation-0')
        agg_path = os.path.join(var_dir, 'aggregate.pt')
        assert os.path.isfile(agg_path)

        # Delete aggregate.pt
        os.remove(agg_path)
        assert not os.path.isfile(agg_path)

        # Epoch 2 -- cache should be rebuilt since aggregate.pt is missing
        _drain(ds)

        assert os.path.isfile(agg_path), \
            "aggregate.pt should be recreated after deletion"


class TestMultipleConcepts:
    """Two concepts with variation grouping should produce separate group dirs."""

    def test_multiple_concepts(self, tmp_path):
        num_items_a = 3
        num_items_b = 3
        total = num_items_a + num_items_b
        tensors_a = _make_tensors(num_items_a, seed=10)
        tensors_b = _make_tensors(num_items_b, seed=20)

        concept_dict_a = {
            'name': 'ConceptA',
            'variations': 1,
            'balancing': 1.0,
            'balancing_strategy': 'REPEATS',
        }
        concept_dict_b = {
            'name': 'ConceptB',
            'variations': 1,
            'balancing': 1.0,
            'balancing_strategy': 'REPEATS',
        }

        # DummyDataModule outputs per-item data: first num_items_a belong to
        # ConceptA, next num_items_b to ConceptB.
        all_tensors = tensors_a + tensors_b
        all_resolutions = [(64, 64)] * total
        all_concepts = [concept_dict_a] * num_items_a + [concept_dict_b] * num_items_b

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[
                {'name': 'ConceptA', 'path': 'pathA'},
                {'name': 'ConceptB', 'path': 'pathB'},
            ],
            dummy_data={
                'latent': all_tensors,
                'crop_resolution': all_resolutions,
                'concept': all_concepts,
            },
            dummy_length=total,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            variations_in_name='concept.variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name='concept.name',
        )

        _drain(ds)

        key_a = _string_key(['ConceptA'])
        key_b = _string_key(['ConceptB'])

        dir_a = os.path.join(cache_dir, key_a, 'variation-0')
        dir_b = os.path.join(cache_dir, key_b, 'variation-0')

        assert os.path.isdir(dir_a), f"Missing group dir for ConceptA: {dir_a}"
        assert os.path.isdir(dir_b), f"Missing group dir for ConceptB: {dir_b}"

        assert os.path.isfile(os.path.join(dir_a, 'aggregate.pt'))
        assert os.path.isfile(os.path.join(dir_b, 'aggregate.pt'))

        # Each group should have the right number of split files
        for i in range(num_items_a):
            assert os.path.isfile(os.path.join(dir_a, f'{i}.pt'))
        for i in range(num_items_b):
            assert os.path.isfile(os.path.join(dir_b, f'{i}.pt'))

        # Verify data integrity per group
        for i in range(num_items_a):
            cached_a = torch.load(os.path.join(dir_a, f'{i}.pt'),
                                  weights_only=False, map_location='cpu')
            assert torch.equal(cached_a['latent'], tensors_a[i])

        for i in range(num_items_b):
            cached_b = torch.load(os.path.join(dir_b, f'{i}.pt'),
                                  weights_only=False, map_location='cpu')
            assert torch.equal(cached_b['latent'], tensors_b[i])


class TestBalancingRepeats:
    """With balancing=0.5 and strategy=REPEATS, half the items should appear."""

    def test_balancing_repeats(self, tmp_path):
        num_items = 6
        tensors = _make_tensors(num_items)

        balancing = 0.5
        expected_output = int(math.floor(num_items * balancing))  # 3

        concept_dict = {
            'name': 'BalRep',
            'variations': 1,
            'balancing': balancing,
            'balancing_strategy': 'REPEATS',
        }

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'BalRep', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
                'concept': [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            variations_in_name='concept.variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name='concept.name',
        )

        batches = _drain(ds)
        assert len(batches) == expected_output, \
            f"Expected {expected_output} items with REPEATS balancing=0.5, got {len(batches)}"


class TestBalancingSamples:
    """With balancing=3 and strategy=SAMPLES, exactly 3 items should appear."""

    def test_balancing_samples(self, tmp_path):
        num_items = 8
        tensors = _make_tensors(num_items)
        target_samples = 3

        concept_dict = {
            'name': 'BalSamp',
            'variations': 1,
            'balancing': float(target_samples),
            'balancing_strategy': 'SAMPLES',
        }

        ds, cache_dir = _build_pipeline(
            tmp_path,
            concepts=[{'name': 'BalSamp', 'path': 'dummy'}],
            dummy_data={
                'latent': tensors,
                'crop_resolution': [(64, 64)] * num_items,
                'concept': [concept_dict] * num_items,
            },
            dummy_length=num_items,
            split_names=['latent'],
            aggregate_names=['crop_resolution'],
            variations_in_name='concept.variations',
            balancing_in_name='concept.balancing',
            balancing_strategy_in_name='concept.balancing_strategy',
            variations_group_in_name='concept.name',
        )

        batches = _drain(ds)
        assert len(batches) == target_samples, \
            f"Expected exactly {target_samples} items with SAMPLES strategy, got {len(batches)}"
