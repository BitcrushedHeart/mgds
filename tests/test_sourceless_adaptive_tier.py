"""Sourceless adaptive tier membership must follow the bucket grid, not sqrt(h*w).

Quantized buckets can exceed their tier's nominal size: the 768 tier holds
640x960 (784 nominal) and 512x1216 / 448x1344 (~789 / ~776). A sourceless run
has no started AspectBucketing, so it used to reject those variants against a
stamped 768 maximum and silently drop the row.
"""

import sys
from pathlib import Path

from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_aspect_bucketing_resolution_eligibility import AspectBucketingHarness  # noqa: E402


def _unstarted_bucketing(quantization=64):
    """The AspectBucketing a sourceless run holds: constructed, never started."""
    return AspectBucketing(
        quantization=quantization,
        resolution_in_name="original_resolution",
        target_resolution_in_name="target_resolution",
        enable_target_resolutions_override_in_name="enable_override",
        target_resolutions_override_in_name="resolution_override",
        target_frames_in_name="target_frames",
        frame_dim_enabled=False,
        scale_resolution_out_name="scale_resolution",
        crop_resolution_out_name="crop_resolution",
        possible_resolutions_out_name="possible_resolutions",
        adaptive_resolution=True,
    )


def _cache(aspect_bucketing, *, adaptive=True, maximum=768):
    cache = SmartDiskCache.__new__(SmartDiskCache)
    cache.aspect_bucketing = aspect_bucketing
    cache.adaptive_resolution = adaptive
    cache.adaptive_resolution_max = maximum
    cache.adaptive_resolution_ladder = None
    cache.additional_resolution_in_names = []
    cache._active_key_by_filepath = {}
    cache.current_variation = 0
    return cache


def _variant(name, crop, target=None):
    record = {"cache_file": name, "crop_resolution": list(crop)}
    if target is not None:
        record["target_resolution"] = target
    return record


def _entry(source, variants, maximum=768):
    return {
        "original_resolution": list(source),
        "source_resolutions": {"original_resolution": list(source)},
        "adaptive_max_resolution": maximum,
        "adaptive_resolution_applied": True,
        "variants": {f"{crop[0]}x{crop[1]}": record for crop, record in variants},
    }


def test_unstarted_reconstruction_matches_started_grid():
    started = AspectBucketingHarness(targets="1024", adaptive_resolution=True)
    unstarted = _unstarted_bucketing()

    assert unstarted.bucket_resolutions == {}
    for target, buckets in started.bucket_resolutions.items():
        assert unstarted.reconstruct_bucket_resolutions(target) == buckets
        for aspect in (0.25, 0.4, 0.5, 2 / 3, 0.8, 1.0, 1.3, 1.5, 2.0, 2.4, 4.0):
            assert unstarted.reconstruct_bucket_for_aspect(aspect, target) == started.bucket_for_aspect(aspect, target)
    # Reconstruction must not make the module look started to SmartDiskCache.
    assert unstarted.bucket_resolutions == {}


@pytest.mark.parametrize("with_bucketing", [True, False])
def test_768_tier_640x960_variant_stays_eligible(with_bucketing):
    # Without a bucketing module the stamped target_resolution is the evidence.
    cache = _cache(_unstarted_bucketing() if with_bucketing else None)
    entry = _entry((1000, 1500), [((640, 960), _variant("tier768", (640, 960), target=768))])

    assert [key for key, _ in cache._training_eligible_variants(entry)] == ["640x960"]
    assert cache._sourceless_variant(entry, in_index=0)["cache_file"] == "tier768"


def test_768_tier_640x960_variant_eligible_without_target_stamp():
    cache = _cache(_unstarted_bucketing())
    entry = _entry((1000, 1500), [((640, 960), _variant("tier768", (640, 960)))])

    assert cache._sourceless_variant(entry, in_index=0)["cache_file"] == "tier768"


@pytest.mark.parametrize(
    "source, crop",
    [
        ((1024, 2432), (512, 1216)),
        ((2432, 1024), (1216, 512)),
        ((900, 2700), (448, 1344)),
        ((2700, 900), (1344, 448)),
    ],
)
def test_768_tier_portrait_and_landscape_extremes_stay_eligible(source, crop):
    cache = _cache(_unstarted_bucketing())
    entry = _entry(source, [(crop, _variant("extreme", crop))])

    assert cache._sourceless_variant(entry, in_index=0)["cache_file"] == "extreme"


@pytest.mark.parametrize("with_bucketing", [True, False])
@pytest.mark.parametrize("stamp", [None, 1024])
def test_variant_from_a_higher_tier_is_still_rejected(with_bucketing, stamp):
    cache = _cache(_unstarted_bucketing() if with_bucketing else None)
    # A large source fits the 1024 bucket, but the entry is capped at 768.
    entry = _entry((2432, 3552), [((832, 1216), _variant("tier1024", (832, 1216), target=stamp))])

    assert cache._training_eligible_variants(entry) == []
    assert cache._sourceless_variant(entry, in_index=0) is None


def test_capped_tier_is_preferred_over_a_cached_higher_tier():
    cache = _cache(_unstarted_bucketing())
    entry = _entry(
        (2000, 3000),
        [
            ((640, 960), _variant("tier768", (640, 960), target=768)),
            ((832, 1280), _variant("tier1024", (832, 1280), target=1024)),
        ],
    )

    assert [key for key, _ in cache._training_eligible_variants(entry)] == ["640x960"]


def test_upscaled_768_tier_variant_is_never_trained():
    cache = _cache(_unstarted_bucketing())
    # 600x900 cannot cover 640x960 and the cached tier is not the floor.
    entry = _entry((600, 900), [((640, 960), _variant("upscaled", (640, 960), target=768))])

    assert cache._training_eligible_variants(entry) == []


def test_unstamped_over_nominal_floor_bucket_is_kept_for_a_tiny_source():
    cache = _cache(_unstarted_bucketing())
    # The 128 tier's 128x192 bucket is 157 nominal; the source fits no tier,
    # so the smallest tier is the intentional (upscaled) fallback.
    entry = _entry((100, 150), [((128, 192), _variant("floor", (128, 192)))])

    assert cache._sourceless_variant(entry, in_index=0)["cache_file"] == "floor"


@pytest.mark.parametrize(
    "source",
    [(1000, 1500), (1500, 1000), (1024, 2432), (2432, 1024), (900, 2700), (800, 800), (700, 1100), (300, 450)],
)
def test_sourceless_selection_matches_sourced_selection(source):
    started = AspectBucketingHarness(targets="768", adaptive_resolution=True)
    aspect = source[0] / source[1]
    # Cache every tier's bucket for this aspect (what sourced builds may have
    # produced across earlier runs), plus a stale higher-tier one.
    crops = {started.bucket_for_aspect(aspect, target) for target in started.bucket_resolutions}
    crops.add(AspectBucketingHarness(targets="1024").bucket_for_aspect(aspect, 1024))
    entry = _entry(source, [(crop, _variant(f"{crop[0]}x{crop[1]}", crop)) for crop in sorted(crops)])

    sourced = _cache(started)._training_eligible_variants(entry)
    sourceless = _cache(_unstarted_bucketing())._training_eligible_variants(entry)

    assert sourced
    assert sourceless == sourced
