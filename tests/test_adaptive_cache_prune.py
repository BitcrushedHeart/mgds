import json

from mgds.pipelineModules.SmartDiskCache import SmartDiskCache


def _write_cache(tmp_path):
    cache_index = {
        "version": 3,
        "entries": {
            "small.png": {
                "original_resolution": [900, 900],
                "source_resolutions": {
                    "original_resolution": [900, 900],
                    "rejected_original_resolution": [900, 900],
                },
                "variants": {
                    "768x768": {"cache_file": "small-safe", "crop_resolution": [768, 768]},
                    "1024x1024": {"cache_file": "shared-high", "crop_resolution": [1024, 1024]},
                },
            },
            "large.png": {
                "original_resolution": [1400, 1400],
                "source_resolutions": {
                    "original_resolution": [1400, 1400],
                    "rejected_original_resolution": [1400, 1400],
                },
                "variants": {
                    "1024x1024": {"cache_file": "shared-high", "crop_resolution": [1024, 1024]},
                },
            },
            "pair.png": {
                "original_resolution": [1400, 1400],
                "source_resolutions": {
                    "original_resolution": [1400, 1400],
                    "rejected_original_resolution": [700, 700],
                },
                "variants": {
                    "512x512": {"cache_file": "pair-safe", "crop_resolution": [512, 512]},
                    "768x768": {"cache_file": "pair-high", "crop_resolution": [768, 768]},
                },
            },
            "floor.png": {
                "original_resolution": [64, 64],
                "source_resolutions": {
                    "original_resolution": [64, 64],
                    "rejected_original_resolution": [64, 64],
                },
                "variants": {
                    "128x128": {"cache_file": "floor", "crop_resolution": [128, 128]},
                },
            },
            "floor-aspect.png": {
                "original_resolution": [64, 128],
                "source_resolutions": {
                    "original_resolution": [64, 128],
                    "rejected_original_resolution": [64, 128],
                },
                "variants": {
                    "128x256": {
                        "cache_file": "floor-aspect",
                        "crop_resolution": [128, 256],
                        "target_resolution": 128,
                    },
                },
            },
            "legacy.png": {
                "variants": {
                    "1024x1024": {"cache_file": "legacy", "crop_resolution": [1024, 1024]},
                },
            },
        },
    }
    index_path = tmp_path / "cache.json"
    index_path.write_text(json.dumps(cache_index), encoding="utf-8")
    for name in [
        "small-safe",
        "shared-high",
        "pair-safe",
        "pair-high",
        "floor",
        "floor-aspect",
        "legacy",
    ]:
        (tmp_path / f"{name}_1.pt").write_bytes(name.encode())
    (tmp_path / "content_index.json").write_text(
        json.dumps({"version": 1, "entries": {"pair": "pair-high_1.pt", "shared": "shared-high_1.pt"}}),
        encoding="utf-8",
    )
    return index_path


def test_adaptive_prune_preview_is_read_only_and_reports_selective_changes(tmp_path):
    index_path = _write_cache(tmp_path)
    before = index_path.read_bytes()

    result = SmartDiskCache.preview_adaptive_prune(
        str(tmp_path), additional_resolution_names=["rejected_original_resolution"]
    )

    assert index_path.read_bytes() == before
    assert result["affected_entries"] == 2
    assert result["removed_variants"] == 2
    assert result["files"] == 1  # shared-high remains referenced by large.png
    assert result["bytes"] == len(b"pair-high")
    assert result["skipped_missing_metadata"] == 1
    assert result["entries_without_usable_variant"] == 0


def test_adaptive_prune_clean_updates_index_then_only_deletes_unreferenced_files(tmp_path):
    index_path = _write_cache(tmp_path)

    result = SmartDiskCache.clean_adaptive_prune(
        str(tmp_path), additional_resolution_names=["rejected_original_resolution"]
    )

    cache_index = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(cache_index["entries"]["small.png"]["variants"]) == {"768x768"}
    assert set(cache_index["entries"]["pair.png"]["variants"]) == {"512x512"}
    assert "128x128" in cache_index["entries"]["floor.png"]["variants"]
    assert "128x256" in cache_index["entries"]["floor-aspect.png"]["variants"]
    assert (tmp_path / "shared-high_1.pt").is_file()
    assert not (tmp_path / "pair-high_1.pt").exists()
    assert not (tmp_path / "cache.json.tmp").exists()
    content_index = json.loads((tmp_path / "content_index.json").read_text(encoding="utf-8"))
    assert content_index["entries"] == {"shared": "shared-high_1.pt"}
    assert result["deleted_files"] == 1


def test_adaptive_prune_can_resolve_legacy_dpo_dimensions_without_tensor_reads(tmp_path):
    index = {
        "entries": {
            "chosen.png": {
                "original_resolution": [1400, 1400],
                "variants": {"768x768": {"cache_file": "pair-high", "crop_resolution": [768, 768]}},
            }
        }
    }
    (tmp_path / "cache.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "pair-high_1.pt").write_bytes(b"cached")

    result = SmartDiskCache.preview_adaptive_prune(
        str(tmp_path),
        additional_resolution_names=["rejected_original_resolution"],
        additional_resolution_provider=lambda _path, _entry: {"rejected_original_resolution": [700, 700]},
    )

    assert result["removed_variants"] == 1
    assert result["skipped_missing_metadata"] == 0


def test_adaptive_prune_preserves_the_configured_floor(tmp_path):
    index = {
        "entries": {
            "tiny.png": {
                "original_resolution": [80, 80],
                "adaptive_max_resolution": 1024,
                "variants": {
                    "192x192": {
                        "cache_file": "custom-floor",
                        "crop_resolution": [192, 192],
                        "target_resolution": 192,
                    }
                },
            }
        }
    }
    (tmp_path / "cache.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "custom-floor_1.pt").write_bytes(b"cached")

    result = SmartDiskCache.preview_adaptive_prune(
        str(tmp_path), adaptive_resolution_ladder="896,640,320,192", adaptive_resolution_max=1024
    )

    assert result["removed_variants"] == 0
    assert result["files"] == 0
