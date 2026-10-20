"""
Status reporting script for HIDS project phases.

For phases 4, 5, and 6 (deep-learning phases), a phase is only marked DONE if:
  1. The key output CSV exists on disk, AND
  2. The corresponding summary.md contains the line "mode: full"

For all other phases, presence of the key file is sufficient.
"""
import sys, os
from datetime import datetime
from pathlib import Path
from config import PHASE_REGISTRY, PROJECT_ROOT

# Phases that require "mode: full" verification in their summary.md
FULL_MODE_REQUIRED = {4, 5, 6}


def _has_full_mode(summary_path: Path) -> bool:
    """Return True if the summary markdown contains 'mode: full'."""
    if not summary_path.is_file():
        return False
    try:
        text = summary_path.read_text(encoding="utf-8", errors="ignore")
        return "mode: full" in text
    except Exception:
        return False


def get_status():
    total_weight = 0
    completed_weight = 0
    rows = []

    for num, info in sorted(PHASE_REGISTRY.items()):
        key_path = PROJECT_ROOT / info["key_file"]
        weight   = info["weight"]
        total_weight += weight

        if key_path.is_file():
            if num in FULL_MODE_REQUIRED:
                # Must also contain "mode: full" in the summary file
                summary_path = PROJECT_ROOT / info.get("summary", "")
                if _has_full_mode(summary_path):
                    status = "DONE (full)"
                    completed_weight += weight
                else:
                    status = "DONE (demo only — rerun without --demo)"
            else:
                status = "DONE"
                completed_weight += weight
            mtime = datetime.fromtimestamp(key_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            status = "PENDING"
            mtime = "-"

        rows.append({
            "phase":    num,
            "title":    info["title"],
            "status":   status,
            "key_file": info["key_file"],
            "date":     mtime,
            "weight":   weight
        })

    pct_complete = (completed_weight / total_weight * 100.0) if total_weight else 0.0
    return rows, pct_complete, completed_weight, total_weight


def print_status_table():
    rows, pct_complete, completed_weight, total_weight = get_status()
    print("\n" + "=" * 100)
    print("                     HOST-BASED INTRUSION DETECTION PROJECT STATUS")
    print("=" * 100)
    print(f"{'Phase':<7s} {'Title':<45s} {'Weight':<8s} {'Status':<30s} {'Last Modified':<18s}")
    print("-" * 100)
    for r in rows:
        print(f"Phase {r['phase']:<1d} {r['title']:<45s} {r['weight']:>3d}%     {r['status']:<30s} {r['date']:<18s}")
        print(f"        -> Output: {r['key_file']}")
    print("-" * 100)
    print(f"Overall Completion: {pct_complete:.1f}% ({completed_weight} / {total_weight} points)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print_status_table()
