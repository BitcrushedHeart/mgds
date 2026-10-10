"""Shrink existing SmartDiskCache ``cache.json`` files in place.

The sourceless cache used to embed a full ~80-field concept config dict into
every (entry, row, variation) runtime-values block, ballooning the index to
multiple GB and making the per-epoch ``json.dumps`` of it the dominant cost
(tens of GB transient RAM, minutes of wall time). The loader now interns each
distinct concept once into ``cache_index['concepts']`` and references it by id;
this tool applies the same one-time migration offline so an already-bloated
cache is fixed without waiting for the next training save.

Usage (run with the trainer STOPPED — it loads the whole index into RAM, which
for a multi-GB cache transiently needs tens of GB, and it rewrites the file):

    # dry run: report projected savings, write nothing
    venv/Scripts/python.exe tools/migrate_intern_concepts.py F:/workspace/SoReal!/cache

    # apply: rewrite each cache.json compactly, keeping a .bak
    venv/Scripts/python.exe tools/migrate_intern_concepts.py F:/workspace/SoReal!/cache --apply

A path may be a cache.json file or a directory (searched recursively for
cache.json). Idempotent: re-running on a migrated cache reports 0 removed.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Single source of truth: reuse the exact migration the loader runs.
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache


def _find_cache_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _dirs, files in os.walk(path):
        if "cache.json" in files:
            found.append(os.path.join(root, "cache.json"))
    return sorted(found)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:,.2f} {unit}"
        n /= 1024
    return f"{n:,.2f} TB"


def migrate_file(path: str, apply: bool) -> None:
    disk_before = os.path.getsize(path)
    print(f"\n{path}")
    print(f"  on disk before : {_human(disk_before)}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    removed = SmartDiskCache._migrate_intern_sourceless_concepts(data)
    n_concepts = len(data.get("concepts", {}))

    # Match the loader's authoritative save format (indent=2) for the size
    # estimate so the projection reflects what actually lands on disk.
    payload = json.dumps(data, indent=2)
    projected = len(payload.encode("utf-8"))

    print(f"  concept copies removed : {removed:,}")
    print(f"  unique concepts        : {n_concepts:,}")
    print(f"  projected after        : {_human(projected)}", end="")
    if projected:
        print(f"  ({disk_before / projected:.1f}x smaller)")
    else:
        print()

    if removed == 0:
        print("  already interned — nothing to write")
        return

    if not apply:
        print("  DRY RUN — pass --apply to rewrite")
        return

    tmp_path = path + ".tmp"
    bak_path = path + ".bak"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
    shutil.copy2(path, bak_path)
    os.replace(tmp_path, path)
    print(f"  written (backup at {os.path.basename(bak_path)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="cache.json file(s) or directory(ies) to migrate")
    ap.add_argument("--apply", action="store_true", help="rewrite the files (default: dry run)")
    args = ap.parse_args()

    targets: list[str] = []
    for p in args.paths:
        targets.extend(_find_cache_files(p))
    if not targets:
        print("No cache.json found under the given paths.")
        return

    print(f"Found {len(targets)} cache.json file(s). Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    for path in targets:
        try:
            migrate_file(path, args.apply)
        except (OSError, json.JSONDecodeError) as e:  # noqa: PERF203 - one-shot tool; isolate per-file failures
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
