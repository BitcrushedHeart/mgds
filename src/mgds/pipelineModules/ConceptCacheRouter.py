"""Route each dataset row to a SmartDiskCache chosen by its concept.

A concept can carry its own cache root (``concept.cache_dir``) so its latents
and text encodings live outside the run's cache directory, e.g. next to a
large dataset that several runs share. SmartDiskCache is a single-directory
store with one cache.json index, so instead of teaching it about several
directories this module partitions the input rows by resolved root and
drives one SmartDiskCache per partition. Each child sees only its own rows
through an index remap and keeps its own index, exactly as if it were the
only cache in the pipeline.

Output order matches a plain SmartDiskCache over the whole dataset: groups are
laid out in the order of their first input row. The downstream
VariationSorting (which serves the uncached names such as prompt and concept)
and the sibling image/text router both assume that order, so every name read
at a given output index still belongs to the same row.
"""

import bisect
import os
from typing import Any

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import SingleVariationRandomAccessPipelineModule


def normalize_cache_root(root: str) -> str:
    """Identity of a cache root: two spellings of one directory are one partition."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(root))))


def resolve_cache_root(override: Any, default_root: str) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()
    return default_root


class _PartitionState:
    """PipelineState view whose sourceless state is private to one partition.

    The image and text caches of a sourceless run share ``anchor_paths`` and the
    group layout through ``sourceless_cache_state``. Each partition has its own
    index, so each needs its own copy; the image and text routers find the same
    copy by partition key. Everything else (the executor) is shared.
    """

    def __init__(self, parent, sourceless_cache_state: dict):
        self._parent = parent
        self.sourceless_cache_state = sourceless_cache_state

    def __getattr__(self, name):
        return getattr(self._parent, name)


class _IndexMappedAspectBucketing:
    """The shared AspectBucketing, seen through a partition's row numbering.

    SmartDiskCache passes its own input indices to the bucketing module to
    reproduce per-row bucket choices; those must be translated back to the
    pipeline's indices. ``_target_override`` writes go to the shared module,
    which is why partitions build one after another, never concurrently.
    """

    _INDEXED = frozenset(
        {
            "_additional_resolutions",
            "_target_resolutions_for_item",
            "variant_key_from_aspect",
            "variant_key_from_resolution",
            "variant_key_from_resolutions",
        }
    )

    def __init__(self, inner, indices: list[int]):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_indices", indices)

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in self._INDEXED:
            return attr
        indices = self._indices

        def mapped(variation, index, *args, **kwargs):
            return attr(variation, indices[index], *args, **kwargs)

        return mapped

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


class _PartitionCache(SmartDiskCache):
    """SmartDiskCache over a subset of the upstream rows."""

    input_indices: list[int] | None = None
    status_label: str | None = None

    def _get_previous_length(self, name: str):
        if self.input_indices is None:
            return super()._get_previous_length(name)
        return len(self.input_indices)

    def _get_previous_item(self, variation: int, name: str, index: int):
        if self.input_indices is not None:
            index = self.input_indices[index]
        return super()._get_previous_item(variation, name, index)

    def _status_label(self) -> str:
        return self.status_label or super()._status_label()


class ConceptCacheRouter(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    def __init__(
        self,
        default_root: str,
        subdir: str,
        cache_kwargs: dict[str, Any],
        root_in_name: str = "concept.cache_dir",
        enabled_in_name: str | None = "concept.enabled",
        sourceless_roots: list[str] | None = None,
    ):
        """
        :param default_root: cache root for rows whose concept sets no override
        :param subdir: directory under each root holding this cache (e.g. "image")
        :param cache_kwargs: SmartDiskCache arguments, minus cache_dir
        :param root_in_name: per-row override root; empty or missing means default
        :param enabled_in_name: rows whose value is falsy are skipped, as SmartDiskCache does
        :param sourceless_roots: roots to serve in sourceless mode, where there are
            no upstream rows to partition. Defaults to just ``default_root``.
        """
        super().__init__()
        self.default_root = default_root
        self.subdir = subdir
        self.cache_kwargs = dict(cache_kwargs)
        self.root_in_name = root_in_name
        self.enabled_in_name = enabled_in_name
        self.sourceless = bool(self.cache_kwargs.get("sourceless", False))
        self.sourceless_roots = list(sourceless_roots) if sourceless_roots else [default_root]

        # Never started; answers get_inputs/get_outputs before any partition exists.
        self._template = SmartDiskCache(cache_dir=os.path.join(default_root, subdir), **self.cache_kwargs)

        self.children: list[_PartitionCache] = []
        self._segments: list[tuple[int, int, _PartitionCache, int]] = []
        self._segment_starts: list[int] = []
        self._child_init_args = None
        self.initialized = False

    def init(self, pipeline, base_seed: int, module_index: int, state):
        super().init(pipeline, base_seed, module_index, state)
        self._child_init_args = (pipeline, base_seed, module_index, state)

    def length(self) -> int:
        if not self.initialized:
            names = self._template.split_names or self._template.aggregate_names
            return self._get_previous_length(names[0])
        return sum(count for _, count, _, _ in self._segments)

    def get_inputs(self) -> list[str]:
        inputs = self._template.get_inputs()
        if not self.sourceless:
            inputs = inputs + [self.root_in_name]
            if self.enabled_in_name and self.enabled_in_name not in inputs:
                inputs = inputs + [self.enabled_in_name]
        return inputs

    def get_outputs(self) -> list[str]:
        return self._template.get_outputs()

    def _partition_rows(self) -> list[tuple[str, str, list[int] | None]]:
        """(root, key, input indices) per partition, in first-row order."""
        if self.sourceless:
            partitions = {}
            for root in self.sourceless_roots:
                partitions.setdefault(normalize_cache_root(root), root)
            return [(root, key, None) for key, root in partitions.items()]

        partitions: dict[str, tuple[str, list[int]]] = {}
        for in_index in range(self._get_previous_length(self.root_in_name)):
            if self.enabled_in_name and not self._get_previous_item(0, self.enabled_in_name, in_index):
                continue
            root = resolve_cache_root(self._get_previous_item(0, self.root_in_name, in_index), self.default_root)
            key = normalize_cache_root(root)
            partitions.setdefault(key, (root, []))[1].append(in_index)
        return [(root, key, indices) for key, (root, indices) in partitions.items()]

    def _make_child(self, root: str, key: str, indices: list[int] | None) -> _PartitionCache:
        kwargs = dict(self.cache_kwargs)
        if indices is not None and kwargs.get("aspect_bucketing") is not None:
            kwargs["aspect_bucketing"] = _IndexMappedAspectBucketing(kwargs["aspect_bucketing"], indices)
        child = _PartitionCache(cache_dir=os.path.join(root, self.subdir), **kwargs)
        child.input_indices = indices
        if key != normalize_cache_root(self.default_root):
            child.status_label = f"{self.subdir}@{os.path.basename(os.path.normpath(root)) or root}"

        pipeline, base_seed, module_index, state = self._child_init_args
        partition_state = state.sourceless_cache_state.setdefault(("ConceptCacheRouter", key), {})
        # Same module index as the router: children walk the modules before
        # it, and their per-epoch SAMPLES shuffles use the router's seeds.
        child.init(pipeline, base_seed, module_index, _PartitionState(state, partition_state))
        return child

    def _init_partitions(self):
        default_key = normalize_cache_root(self.default_root)
        self.children = []
        for root, key, indices in self._partition_rows():
            if key != default_key:
                rows = "sourceless" if indices is None else f"{len(indices)} rows"
                print(f"Concept cache override ({self.subdir}): {rows} -> {os.path.join(root, self.subdir)}")
            self.children.append(self._make_child(root, key, indices))

    def _build_layout(self):
        segments = []
        for child_order, child in enumerate(self.children):
            offset = 0
            for group_key, count in child.group_output_samples.items():
                if count > 0:
                    if child.input_indices is None:
                        first_row = 0
                    else:
                        first_row = child.input_indices[child.group_full_indices[group_key][0]]
                    segments.append((first_row, child_order, count, child, offset))
                offset += count

        # Sourceless partitions have no shared row numbering; they concatenate
        # in root order, which the sibling router shares.
        if not self.sourceless:
            segments.sort(key=lambda segment: (segment[0], segment[1]))

        self._segments = []
        self._segment_starts = []
        start = 0
        for _first_row, _child_order, count, child, offset in segments:
            self._segments.append((start, count, child, offset))
            self._segment_starts.append(start)
            start += count

    def start(self, variation: int):
        if not self.initialized:
            self._init_partitions()
        for child in self.children:
            child.current_variation = variation
            child.clear_item_cache()
            child.start(variation)
        self._build_layout()
        self.initialized = True

    def get_item(self, index: int, requested_name: str = None) -> dict:
        position = bisect.bisect_right(self._segment_starts, index) - 1
        if position >= 0:
            start, count, child, offset = self._segments[position]
            if index < start + count:
                return child.get_item(offset + index - start, requested_name)
        return {requested_name: self._get_previous_item(self.current_variation, requested_name, index)}
