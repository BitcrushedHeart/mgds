import concurrent
import hashlib
import json
import math
import os
import pathlib
from typing import Any, Callable

import torch
from tqdm import tqdm

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule
from mgds.util.PersistentCacheState import PersistentCacheState
from mgds.util.FileUtil import safe_write_json_file

def _clone_for_cache(item: Any) -> Any:
    if isinstance(item, torch.Tensor):
        return item.clone()
    else:
        return item

def _build_cache_data(pipeline: PipelineModule,
                      persistent_cache: PersistentCacheState,
                      split_names: list[str],
                      aggregate_names: list[str],
                      group_index: int,
                      in_index: int,
                      in_variation: int,
                      current_thread_cuda_device: int) -> tuple[int, dict]:
    # preserve current device for multi-GPU, which is thread-local in torch:
    if torch.cuda.is_available() and current_thread_cuda_device is not None:
        torch.cuda.set_device(current_thread_cuda_device)

    with torch.no_grad():
        split_item = {}
        for name in split_names:
            split_item[name] = _clone_for_cache(pipeline._get_previous_item(in_variation, name, in_index))
        persistent_cache.save_split_item(group_index, split_item)
        del split_item

        aggregate_item = {}
        for name in aggregate_names:
            aggregate_item[name] = _clone_for_cache(pipeline._get_previous_item(in_variation, name, in_index))
        return group_index, aggregate_item


class DiskCache(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
            self,
            cache_dir: str,
            persistent_key_in_name: str | None = None,
            split_names: list[str] | None = None,
            aggregate_names: list[str] | None = None,
            variations_in_name: str | None = None,
            balancing_in_name: str | None = None,
            balancing_strategy_in_name: str | None = None,
            variations_group_in_name: str | list[str] | None = None,
            group_enabled_in_name: str | None = None,
            group_name_in_name: str | None = None,
            before_cache_fun: Callable[[], None] | None = None,
            allow_unsafe_types: bool = False,
    ):
        super(DiskCache, self).__init__()
        self.cache_dir = cache_dir

        self.persistent_key_in_name = persistent_key_in_name

        self.split_names = [] if split_names is None else split_names
        self.aggregate_names = [] if aggregate_names is None else aggregate_names

        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_names = \
            [variations_group_in_name] if isinstance(variations_group_in_name, str) else variations_group_in_name

        self.group_enabled_in_name = group_enabled_in_name
        self.group_name_in_name = group_name_in_name

        self.before_cache_fun = (lambda: None) if before_cache_fun is None else before_cache_fun

        self.allow_unsafe_types = allow_unsafe_types

        self.group_variations = {}
        self.group_indices = {}
        self.group_output_samples = {}
        self.variations_initialized = False

    def length(self) -> int:
        if not self.variations_initialized:
            name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]
            return self._get_previous_length(name)
        else:
            return sum(x for x in self.group_output_samples.values())

    def get_inputs(self) -> list[str]:
        return self.split_names + self.aggregate_names \
            + [self.variations_in_name] if self.variations_in_name else [] \
            + [self.balancing_in_name] if self.variations_in_name else [] \
            + [self.balancing_strategy_in_name] if self.variations_in_name else [] \
            + self.variations_group_in_names if self.variations_in_name else [] \
            + [self.group_enabled_in_name] if self.variations_in_name else []

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names

    def _get_persistent_key(self, variation: int, index: int) -> str | None:
        """
        Look up the persistent key (e.g. image_path) for a given index by walking
        backward through the pipeline, skipping other DiskCache modules.

        When multiple DiskCache modules are chained (e.g. image cache -> text cache),
        a downstream cache's in_index values come from the original data source
        (CollectPaths), not from the upstream cache's output space. The upstream cache
        may have a different (smaller) output length due to balancing, causing
        IndexError when the downstream cache tries to look up persistent keys through
        the upstream cache. By skipping DiskCache modules, we reach the original data
        source directly.
        """
        if self.persistent_key_in_name is None:
            return None

        split_name = self.persistent_key_in_name.split('.')
        item_name = split_name[0]
        path_names = split_name[1:]

        # Walk backward through pipeline modules, skipping other DiskCache instances
        # noinspection PyUnresolvedReferences
        for prev_idx in range(self._PipelineModule__module_index - 1, -1, -1):
            module = self.pipeline.modules[prev_idx]

            if item_name not in module.get_outputs():
                continue

            # Skip other DiskCache modules — they have a different index space
            if isinstance(module, DiskCache):
                continue

            try:
                if isinstance(module, RandomAccessPipelineModule):
                    item = module.get_item(variation, index, item_name)
                elif isinstance(module, SingleVariationRandomAccessPipelineModule):
                    item = module.get_item(index, item_name)
                else:
                    continue

                if item is None:
                    continue

                result = item[item_name]
                for path_name in path_names:
                    if isinstance(result, dict) and path_name in result:
                        result = result[path_name]
                    else:
                        result = None
                        break
                if result is not None:
                    return result
            except (IndexError, KeyError):
                continue

        return None

    def __string_key(self, data: list[Any]) -> str:
        json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'), indent=None)
        return hashlib.sha256(json_data.encode('utf-8')).hexdigest()

    def __init_variations(self):
        """
        Prepares variations before caching starts. Each index is sorted into a group.

        Data is written into three variables.
            self.group_variations, mapping group keys to the number of variations of that group
            self.group_indices, mapping group keys to a list of input indices contained in the group
            self.group_output_samples, mapping group keys to the number of indices in the cache output for each group
        """
        if self.variations_in_name is not None:
            group_variations: dict[str, int] = {}
            group_indices: dict[str, list] = {}
            group_balancing: dict[str, int] = {}
            group_balancing_strategy: dict[str, str] = {}

            for in_index in range(self._get_previous_length(self.variations_in_name)):
                if self.group_enabled_in_name and not self._get_previous_item(0, self.group_enabled_in_name, in_index):
                    continue

                variations = self._get_previous_item(0, self.variations_in_name, in_index)
                balancing = self._get_previous_item(0, self.balancing_in_name, in_index)
                balancing_strategy = self._get_previous_item(0, self.balancing_strategy_in_name, in_index)
                group_key = self.__string_key(
                    [self._get_previous_item(0, name, in_index) for name in self.variations_group_in_names]
                )

                if group_key not in group_variations:
                    group_variations[group_key] = variations

                if group_key not in group_indices:
                    group_indices[group_key] = []
                group_indices[group_key].append(in_index)

                if group_key not in group_balancing:
                    group_balancing[group_key] = balancing

                if group_key not in group_balancing_strategy:
                    group_balancing_strategy[group_key] = balancing_strategy

            group_output_samples = {}
            for group_key, balancing in group_balancing.items():
                balancing_strategy = group_balancing_strategy[group_key]
                if balancing_strategy == 'REPEATS':
                    group_output_samples[group_key] = int(math.floor(len(group_indices[group_key]) * balancing))
                if balancing_strategy == 'SAMPLES':
                    group_output_samples[group_key] = int(balancing)
        else:
            first_previous_name = self.split_names[0] if len(self.split_names) > 0 else self.aggregate_names[0]

            group_variations = {'': 1}
            group_indices = {'': [in_index for in_index in range(self._get_previous_length(first_previous_name))]}
            group_output_samples = {'': len(group_indices[''])}

        self.aggregate_cache = {}
        self.persistent_cache = {}

        self.group_variations = group_variations
        self.group_indices = group_indices
        self.group_output_samples = group_output_samples

        self.variations_initialized = True

    def __get_cache_dir_path(self, group_key: str, in_variation: int) -> pathlib.Path:
        variations = self.group_variations[group_key]
        return pathlib.Path(self.cache_dir) / group_key / f"variation-{str(in_variation % variations)}"

    def __refresh_cache(self, out_variation: int):
        if not self.variations_initialized:
            self.__init_variations()

        self.aggregate_cache = {}
        for group_key, variations in self.group_variations.items():
            self.aggregate_cache[group_key] = [None] * variations
            if group_key not in self.persistent_cache:
                self.persistent_cache[group_key] = [None] * variations

        before_cache_fun_called = False
        for group_key in self.group_variations.keys():
            start_index = self.group_output_samples[group_key] * out_variation
            end_index = self.group_output_samples[group_key] * (out_variation + 1) - 1

            start_variation = start_index // len(self.group_indices[group_key])
            end_variation = end_index // len(self.group_indices[group_key])

            variations = self.group_variations[group_key]
            for in_variation in [(x % variations) for x in range(start_variation, end_variation + 1, 1)]:
                cache_path = self.__get_cache_dir_path(group_key, in_variation)

                # Load Persistent Cache State
                persistent_cache_path = cache_path / 'cache_state.json'

                persistent_cache = None
                if persistent_cache_path.is_file():
                    try:
                        with open(persistent_cache_path, 'r') as persistent_cache_reader:
                            persistent_cache_dict = json.load(persistent_cache_reader)
                        persistent_cache = PersistentCacheState.from_dict(cache_path, persistent_cache_dict)
                    except ValueError as ex:
                        print(f'Could not load "{persistent_cache_path.name}" due to error: {str(ex)}.'
                               ' Cache will be rebuilt.')

                # Start a new persistent cache if we failed to load the existing one, or one did not
                # already exist.
                if persistent_cache is None:
                    persistent_cache = PersistentCacheState(cache_path,
                                                            aggregate_filename='aggregate.pt')

                self.persistent_cache[group_key][in_variation] = persistent_cache

                # Initialize Persistent Cache State with the current run's set of files
                cache_stable = self.persistent_key_in_name is not None
                persistent_key_metadata = None
                if cache_stable:
                    # Map each `group_index` to a persistent key--a value that will always be the
                    # same for the file (or the file's data in a perfect world), so that our
                    # persistent cache has a stable value to reference this file by.
                    group_index_mappings = {}
                    for group_index, in_index in enumerate(self.group_indices[group_key]):
                        key = self._get_persistent_key(out_variation, in_index)
                        if key is None:
                            # Fall back to unstable index key; this item will be
                            # re-cached because the key won't match any prior entry.
                            key = str(group_index)
                        group_index_mappings[group_index] = key

                    # Compute file metadata (mtime + size) for content change detection
                    persistent_key_metadata = {}
                    for persistent_key in group_index_mappings.values():
                        if persistent_key not in persistent_key_metadata:
                            try:
                                stat = os.stat(persistent_key)
                                persistent_key_metadata[persistent_key] = f"{stat.st_mtime_ns}:{stat.st_size}"
                            except OSError:
                                persistent_key_metadata[persistent_key] = ""
                else:
                    # As we don't have a persistent key to map each group_index to the file that it
                    # represents, we fall-back to the legacy unstable behavior by mapping each
                    # `group_index` it to a string of itself. This is not "safe", in that any
                    # inclusions/removals to the files on disk will disrupt this order the next time
                    # we run. It is much better to use the code path above this one, but this path
                    # exists for legacy callers that have not been updated.
                    print('Warning: Using unstable cache file mappings.')
                    group_index_mappings = {group_index: str(group_index)
                                            for group_index in range(len(self.group_indices[group_key]))}

                persistent_cache.build_cache_file_mappings(group_index_mappings,
                                                           persistent_key_metadata=persistent_key_metadata,
                                                           remove_stale_cache=False)

                # Resave our cache state if we are using stable file mappings
                if cache_stable:
                    persistent_cache_dict = persistent_cache.to_dict()
                    safe_write_json_file(persistent_cache_dict, persistent_cache_path)

                # Load our aggregate items cache
                aggregate_cache = persistent_cache.get_aggregate_cache(self.pipeline.device,
                                                                       self.allow_unsafe_types,
                                                                       validate_against_split_items=True)
                if aggregate_cache is None:
                    aggregate_cache = [None] * len(self.group_indices[group_key]) 

                self.aggregate_cache[group_key][in_variation] = aggregate_cache
                
                # If we have uncached items, cache them now
                total_uncached_files = aggregate_cache.count(None)
                if total_uncached_files > 0:
                    if not before_cache_fun_called and self.before_cache_fun is not None:
                        before_cache_fun_called = True
                        self.before_cache_fun()

                    total_files = len(aggregate_cache)
                    total_cached_files = total_files - total_uncached_files

                    group_name = (self._get_previous_item(out_variation, self.group_name_in_name, self.group_indices[group_key][0])
                                  if self.group_name_in_name is not None else
                                  None)
                    caching_text = (f'caching {group_name} ({group_key[0:7]}...{group_key[-7:]})'
                                    if group_name is not None else ''\
                                    f'caching {group_key}')

                    with tqdm(initial=total_cached_files, total=total_files, smoothing=0.1, desc=caching_text) as bar:
                        current_thread_cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
                        fs = (self._state.executor.submit(_build_cache_data,
                                                          self,
                                                          persistent_cache,
                                                          self.split_names,
                                                          self.aggregate_names,
                                                          group_index,
                                                          in_index,
                                                          in_variation,
                                                          current_thread_cuda_device)
                              for (group_index, in_index)
                              in enumerate(self.group_indices[group_key])
                              if aggregate_cache[group_index] is None)
                        for i, f in enumerate(concurrent.futures.as_completed(fs)):
                            try:
                                group_index, index_aggregate_data = f.result()
                                aggregate_cache[group_index] = index_aggregate_data
                            except:
                                self._state.executor.shutdown(wait=True, cancel_futures=True)
                                raise

                            if i % 250 == 0:
                                self._torch_gc()
                            if i % 200 == 0:
                                persistent_cache.save_aggregate_cache(aggregate_cache)
                            bar.update(1)

                    persistent_cache.save_aggregate_cache(aggregate_cache)

    def __get_input_index(self, out_variation: int, out_index: int) -> tuple[str, int, int, int]:
        offset = 0
        for group_key, group_output_samples in self.group_output_samples.items():
            if out_index >= group_output_samples + offset:
                offset += group_output_samples
                continue

            variations = self.group_variations[group_key]
            local_index = (out_index - offset) + (out_variation * self.group_output_samples[group_key])
            in_variation = (local_index // len(self.group_indices[group_key])) % variations
            group_index = local_index % len(self.group_indices[group_key])
            in_index = self.group_indices[group_key][group_index]

            return group_key, in_variation, group_index, in_index

    def start(self, out_variation: int):
        self.__refresh_cache(out_variation)

    def get_item(self, index: int, requested_name: str = None) -> dict:
        item = {}

        result = self.__get_input_index(self.current_variation, index)
        if result is None:
            total = sum(self.group_output_samples.values())
            raise IndexError(
                f"DiskCache index {index} out of range (total output samples: {total}, "
                f"current_variation: {self.current_variation})"
            )
        group_key, in_variation, group_index, in_index = result

        if requested_name is None or requested_name in self.aggregate_names:
            variation_cache = self.aggregate_cache[group_key][in_variation]
            if variation_cache is None:
                raise RuntimeError(
                    f"DiskCache: aggregate cache for group {group_key[:14]} variation "
                    f"{in_variation} is None (not loaded). current_variation="
                    f"{self.current_variation}, index={index}"
                )
            aggregate_item: dict[str, Any] = variation_cache[group_index]
            if aggregate_item is None:
                raise RuntimeError(
                    f"DiskCache: aggregate cache entry is None at group_index={group_index} "
                    f"(group={group_key[:14]}, variation={in_variation}). This may indicate "
                    f"a corrupted or partially-written cache. Try deleting the cache directory."
                )
            item = aggregate_item.copy()

        if requested_name is None or requested_name in self.split_names:
            persistent_cache: PersistentCacheState = self.persistent_cache[group_key][in_variation]
            split_item: dict[str, Any] = \
                persistent_cache.get_split_item(group_index,
                                                self.pipeline.device,
                                                allow_unsafe_types=self.allow_unsafe_types)
            for name in self.split_names:
                item[name] = split_item[name]

        return item
