from random import Random

from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.ResolutionPolicy import adaptive_resolution_tiers
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

import numpy as np
import pytest


class AspectBucketingHarness(AspectBucketing):
    def __init__(
        self,
        resolution=(1024, 1024),
        targets="1024,1280",
        *,
        adaptive_resolution=False,
        adaptive_resolution_ladder=None,
        additional_resolutions=None,
        override_enabled=False,
        override_targets=None,
    ):
        super().__init__(
            quantization=64,
            resolution_in_name="original_resolution",
            target_resolution_in_name="target_resolution",
            enable_target_resolutions_override_in_name="enable_override",
            target_resolutions_override_in_name="resolution_override",
            target_frames_in_name="target_frames",
            frame_dim_enabled=False,
            scale_resolution_out_name="scale_resolution",
            crop_resolution_out_name="crop_resolution",
            possible_resolutions_out_name="possible_resolutions",
            adaptive_resolution=adaptive_resolution,
            additional_resolution_in_names=["rejected_original_resolution"] if additional_resolutions else None,
            adaptive_resolution_ladder=adaptive_resolution_ladder,
        )
        self.resolution = resolution
        self.targets = targets
        self.additional_resolutions = list(additional_resolutions or [])
        self.override_enabled = override_enabled
        self.override_targets = override_targets or targets

        target_values = (
            adaptive_resolution_tiers(targets, adaptive_resolution_ladder)
            if adaptive_resolution
            else [int(value) for value in targets.split(",")]
        )
        if adaptive_resolution and self.override_enabled:
            target_values.extend(adaptive_resolution_tiers(self.override_targets, adaptive_resolution_ladder))
        target_values = sorted(set(target_values))
        create_buckets = self._AspectBucketing__create_automatic_buckets
        self.bucket_resolutions, self.bucket_aspects = create_buckets(target_values)
        self.bucket_aspects = {key: np.array(value) for key, value in self.bucket_aspects.items()}
        self.flattened_possible_resolutions = list(set(sum(self.bucket_resolutions.values(), [])))

    def _get_previous_item(self, variation, name, index):
        values = {
            "original_resolution": self.resolution,
            "target_resolution": self.targets,
            "rejected_original_resolution": self.additional_resolutions[0]
            if self.additional_resolutions
            else self.resolution,
            "enable_override": self.override_enabled,
            "resolution_override": self.override_targets,
            "target_frames": "1",
        }
        return values[name]

    def _get_rand(self, variation, index):
        return Random(variation * 1_000_003 + index)


def _selected_keys(bucketing, variations=64):
    return {
        bucketing.variant_key_from_resolution(variation, 7, bucketing.resolution) for variation in range(variations)
    }


def test_image_between_targets_only_selects_lower_resolution():
    bucketing = AspectBucketingHarness(resolution=(1200, 1200))

    assert bucketing.eligible_target_resolutions((1200, 1200), [1024, 1280]) == [1024]
    assert _selected_keys(bucketing) == {"1024x1024"}

    for variation in range(16):
        item = bucketing.get_item(variation, 7)
        assert item["crop_resolution"] == (1024, 1024)
        assert item["scale_resolution"] == (1024, 1024)


def test_image_large_enough_for_both_targets_rotates_between_both():
    bucketing = AspectBucketingHarness(resolution=(1400, 1400))

    assert bucketing.eligible_target_resolutions((1400, 1400), [1024, 1280]) == [1024, 1280]
    assert _selected_keys(bucketing) == {"1024x1024", "1280x1280"}


def test_image_smaller_than_every_target_falls_back_to_smallest_only():
    bucketing = AspectBucketingHarness(resolution=(768, 768), targets="1280,1024")

    assert bucketing.eligible_target_resolutions((768, 768), [1280, 1024]) == [1024]
    assert _selected_keys(bucketing) == {"1024x1024"}


def test_non_square_eligibility_uses_the_aspect_matched_crop_bucket():
    bucketing = AspectBucketingHarness(resolution=(960, 1440))

    assert bucketing.eligible_target_resolutions((960, 1440), [1024, 1280]) == [1024]
    expected_bucket = bucketing.bucket_for_aspect(960 / 1440, 1024)
    assert _selected_keys(bucketing) == {f"{expected_bucket[0]}x{expected_bucket[1]}"}


def test_adaptive_resolution_tiers_use_largest_scalar_and_standard_ladder():
    assert adaptive_resolution_tiers("768, 1024") == [128, 256, 512, 768, 1024]
    assert adaptive_resolution_tiers("900") == [128, 256, 512, 768, 900]
    assert adaptive_resolution_tiers("96") == [96]


def test_adaptive_resolution_tiers_use_configured_snap_ladder():
    ladder = "896, 640, 320, 192"

    assert adaptive_resolution_tiers("1024", ladder) == [192, 320, 640, 896, 1024]
    assert adaptive_resolution_tiers("768", ladder) == [192, 320, 640, 768]


def test_adaptive_resolution_tiers_reject_an_empty_or_invalid_ladder():
    for ladder in ("", "0,-128", "768,nope,128"):
        with pytest.raises(ValueError):
            adaptive_resolution_tiers("1024", ladder)


def test_adaptive_resolution_leaves_exact_dimensions_unchanged():
    assert adaptive_resolution_tiers("1024x768") == []


def test_adaptive_mode_selects_only_highest_fitting_tier():
    cases = [
        ((1400, 1400), (1024, 1024)),
        ((900, 900), (768, 768)),
        ((700, 700), (512, 512)),
        ((400, 400), (256, 256)),
        ((200, 200), (128, 128)),
        ((96, 96), (128, 128)),
    ]

    for source, expected in cases:
        bucketing = AspectBucketingHarness(
            resolution=source,
            targets="1024",
            adaptive_resolution=True,
        )
        selected = {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(32)}
        assert selected == {expected}


def test_adaptive_mode_uses_largest_concept_override_as_maximum():
    bucketing = AspectBucketingHarness(
        resolution=(900, 900),
        targets="1024",
        adaptive_resolution=True,
        override_enabled=True,
        override_targets="512,768",
    )

    assert {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(16)} == {(768, 768)}


def test_adaptive_mode_snaps_to_highest_fitting_custom_tier():
    bucketing = AspectBucketingHarness(
        resolution=(700, 700),
        targets="1024",
        adaptive_resolution=True,
        adaptive_resolution_ladder="896,640,320,192",
    )

    assert {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(16)} == {(640, 640)}


def test_adaptive_mode_uses_the_custom_smallest_tier_as_its_fallback():
    bucketing = AspectBucketingHarness(
        resolution=(100, 100),
        targets="1024",
        adaptive_resolution=True,
        adaptive_resolution_ladder="896,640,320,192",
    )

    assert {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(16)} == {(192, 192)}


def test_custom_ladder_is_capped_by_the_concept_override_maximum():
    bucketing = AspectBucketingHarness(
        resolution=(650, 650),
        targets="1024",
        adaptive_resolution=True,
        adaptive_resolution_ladder="896,640,320,192",
        override_enabled=True,
        override_targets="700",
    )

    assert {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(16)} == {(640, 640)}


def test_dpo_pair_uses_highest_tier_supported_by_both_images():
    bucketing = AspectBucketingHarness(
        resolution=(1400, 1400),
        targets="1024",
        adaptive_resolution=True,
        additional_resolutions=[(700, 700)],
    )

    assert {bucketing.get_item(variation, 7)["crop_resolution"] for variation in range(16)} == {(512, 512)}


def test_low_quantization_never_generates_zero_sized_adaptive_buckets():
    bucketing = AspectBucketingHarness(
        resolution=(128, 512),
        targets="128",
        adaptive_resolution=True,
    )

    assert all(h > 0 and w > 0 for h, w in bucketing.flattened_possible_resolutions)


class SingleAspectHarness(SingleAspectCalculation):
    def __init__(
        self, resolution, targets="1024,1280", *, adaptive=False, rejected=None, adaptive_resolution_ladder=None
    ):
        super().__init__(
            resolution_in_name="original_resolution",
            target_resolution_in_name="target_resolution",
            enable_target_resolutions_override_in_name="enable_override",
            target_resolutions_override_in_name="resolution_override",
            scale_resolution_out_name="scale_resolution",
            crop_resolution_out_name="crop_resolution",
            possible_resolutions_out_name="possible_resolutions",
            adaptive_resolution=adaptive,
            additional_resolution_in_names=["rejected_original_resolution"] if rejected else None,
            adaptive_resolution_ladder=adaptive_resolution_ladder,
        )
        self.resolution = resolution
        self.targets = targets
        self.rejected = rejected or resolution

    def _get_previous_item(self, variation, name, index):
        values = {
            "original_resolution": self.resolution,
            "target_resolution": self.targets,
            "rejected_original_resolution": self.rejected,
            "enable_override": False,
            "resolution_override": "1024,1280",
        }
        return values[name]

    def _get_rand(self, variation, index):
        return Random(variation * 1_000_003 + index)


def test_single_aspect_multires_applies_the_same_no_upscale_rule():
    between = SingleAspectHarness((1200, 1400))
    undersized = SingleAspectHarness((768, 900))
    large = SingleAspectHarness((1400, 1400))

    assert {between.get_item(variation, 5)["crop_resolution"] for variation in range(32)} == {(1024, 1024)}
    assert {undersized.get_item(variation, 5)["crop_resolution"] for variation in range(32)} == {(1024, 1024)}
    assert {large.get_item(variation, 5)["crop_resolution"] for variation in range(64)} == {(1024, 1024), (1280, 1280)}


def test_single_aspect_adaptive_selects_highest_tier_supported_by_dpo_pair():
    calculation = SingleAspectHarness((1400, 1400), targets="1024", adaptive=True, rejected=(700, 900))

    assert {calculation.get_item(variation, 5)["crop_resolution"] for variation in range(32)} == {(512, 512)}


def test_single_aspect_adaptive_uses_custom_ladder():
    calculation = SingleAspectHarness(
        (700, 700), targets="1024", adaptive=True, adaptive_resolution_ladder="896,640,320,192"
    )

    assert {calculation.get_item(variation, 5)["crop_resolution"] for variation in range(16)} == {(640, 640)}


def _cache_without_pipeline(aspect_bucketing=None, *, adaptive=False):
    cache = SmartDiskCache.__new__(SmartDiskCache)
    cache.aspect_bucketing = aspect_bucketing
    cache.adaptive_resolution = adaptive
    cache.additional_resolution_in_names = []
    cache._active_key_by_filepath = {}
    cache.current_variation = 0
    return cache


def _entry(source_resolution):
    return {
        "original_resolution": list(source_resolution),
        "variants": {
            "1024x1024": {
                "cache_file": "low",
                "crop_resolution": [1024, 1024],
            },
            "1280x1280": {
                "cache_file": "high",
                "crop_resolution": [1280, 1280],
            },
        },
    }


def test_cache_fallback_cannot_serve_a_stale_ineligible_high_variant():
    bucketing = AspectBucketingHarness(resolution=(1200, 1200))
    cache = _cache_without_pipeline(bucketing)
    cache._active_key_by_filepath["image.png"] = "1280x1280"

    assert cache._active_cache_file("image.png", _entry((1200, 1200))) == "low"


def test_missing_required_low_variant_does_not_fall_back_to_high_variant():
    bucketing = AspectBucketingHarness(resolution=(768, 768))
    cache = _cache_without_pipeline(bucketing)
    entry = _entry((768, 768))
    del entry["variants"]["1024x1024"]

    assert cache._active_cache_file("image.png", entry) is None


def test_sourceless_rotation_filters_old_ineligible_variants():
    cache = _cache_without_pipeline()
    entry = _entry((1200, 1200))

    selected = set()
    for epoch in range(16):
        cache.current_variation = epoch
        selected.add(cache._sourceless_variant(entry, in_index=3)["cache_file"])

    assert selected == {"low"}


def test_sourceless_undersized_fallback_uses_smallest_cached_tier():
    cache = _cache_without_pipeline()
    entry = _entry((768, 768))

    selected = set()
    for epoch in range(16):
        cache.current_variation = epoch
        selected.add(cache._sourceless_variant(entry, in_index=3)["cache_file"])

    assert selected == {"low"}


def test_cache_adaptive_dpo_never_serves_tier_rejected_image_cannot_support():
    bucketing = AspectBucketingHarness(
        resolution=(1400, 1400),
        targets="1024",
        adaptive_resolution=True,
        additional_resolutions=[(700, 700)],
    )
    cache = _cache_without_pipeline(bucketing, adaptive=True)
    cache.additional_resolution_in_names = ["rejected_original_resolution"]
    cache._active_key_by_filepath["pair.png"] = "768x768"
    entry = {
        "original_resolution": [1400, 1400],
        "source_resolutions": {
            "original_resolution": [1400, 1400],
            "rejected_original_resolution": [700, 700],
        },
        "variants": {
            "512x512": {"cache_file": "safe", "crop_resolution": [512, 512]},
            "768x768": {"cache_file": "too-large", "crop_resolution": [768, 768]},
        },
    }

    assert cache._active_cache_file("pair.png", entry) == "safe"

    assert cache._fast_resolution_string(entry, 0, 7) == "512x512"


def test_sourceless_adaptive_selects_highest_safe_cached_tier():
    cache = _cache_without_pipeline(adaptive=True)
    entry = {
        "original_resolution": [900, 900],
        "variants": {
            "512x512": {"cache_file": "mid", "crop_resolution": [512, 512]},
            "768x768": {"cache_file": "high-safe", "crop_resolution": [768, 768]},
            "1024x1024": {"cache_file": "upscaled", "crop_resolution": [1024, 1024]},
        },
    }

    assert cache._sourceless_variant(entry, in_index=3)["cache_file"] == "high-safe"


def test_cache_adaptive_selection_is_identical_without_aspect_bucketing():
    calculation = SingleAspectHarness((900, 900), targets="1024", adaptive=True)
    calculation.possible_target_resolutions = adaptive_resolution_tiers("1024")
    calculation.bucket_resolutions = {target: [(target, target)] for target in calculation.possible_target_resolutions}
    cache = _cache_without_pipeline(calculation, adaptive=True)
    cache._active_key_by_filepath["image.png"] = "1024x1024"
    entry = {
        "original_resolution": [900, 900],
        "source_resolutions": {"original_resolution": [900, 900]},
        "variants": {
            "768x768": {"cache_file": "safe", "crop_resolution": [768, 768]},
            "1024x1024": {"cache_file": "upscaled", "crop_resolution": [1024, 1024]},
        },
    }

    assert cache._active_cache_file("image.png", entry) == "safe"


def test_exact_wxh_cache_variant_remains_unchanged_when_toggle_is_on():
    cache = _cache_without_pipeline(adaptive=True)
    cache._active_key_by_filepath["image.png"] = "1024x768"
    entry = {
        "original_resolution": [500, 500],
        "adaptive_resolution_applied": False,
        "variants": {
            "1024x768": {"cache_file": "exact", "crop_resolution": [1024, 768]},
        },
    }

    assert cache._active_cache_file("image.png", entry) == "exact"


def test_sourceless_adaptive_drops_entry_without_source_dimensions():
    cache = _cache_without_pipeline(adaptive=True)
    entry = {
        "variants": {
            "768x768": {"cache_file": "unknown", "crop_resolution": [768, 768]},
        }
    }

    assert cache._sourceless_variant(entry, in_index=0) is None


def test_sourceless_adaptive_preserves_non_square_cached_128_floor():
    cache = _cache_without_pipeline(adaptive=True)
    entry = {
        "original_resolution": [64, 128],
        "adaptive_max_resolution": 1024,
        "variants": {
            "128x256": {
                "cache_file": "floor",
                "crop_resolution": [128, 256],
                "target_resolution": 128,
            },
        },
    }

    assert cache._sourceless_variant(entry, in_index=0)["cache_file"] == "floor"


@pytest.mark.parametrize("resolution", AspectBucketing.all_possible_input_resolutions)
@pytest.mark.parametrize("flip", [False, True])
def test_native_generator_resolutions_land_in_exact_buckets(resolution, flip):
    if flip:
        resolution = (resolution[1], resolution[0])
    bucketing = AspectBucketingHarness(resolution=resolution, targets="1024")

    assert bucketing.bucket_for_aspect(resolution[0] / resolution[1], 1024) == resolution
    assert bucketing.eligible_target_resolutions(resolution, [1024]) == [1024]
    item = bucketing.get_item(0, 7)
    assert item["crop_resolution"] == resolution
    assert item["scale_resolution"] == resolution


def test_native_generator_resolutions_stay_eligible_below_a_larger_tier():
    bucketing = AspectBucketingHarness(resolution=(1536, 640), targets="1024,1280")

    assert bucketing.eligible_target_resolutions((1536, 640), [1024, 1280]) == [1024]
    assert _selected_keys(bucketing) == {"1536x640"}
