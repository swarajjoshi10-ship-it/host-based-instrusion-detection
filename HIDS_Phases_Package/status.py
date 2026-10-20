"""
Status reporting script for HIDS project phases.
Computes overall progress ONLY from phases whose key output files actually exist on disk.
"""
import sys, os
from datetime import datetime
from pathlib import Path
from config import PHASE_REGISTRY, PROJECT_ROOT

def get_status():
    total_weight = 0
    completed_weight = 0
    rows = []

    for num, info in sorted(PHASE_REGISTRY.items()):
        key_path = PROJECT_ROOT / info["key_file"]
        weight = info["weight"]
        total_weight += weight
        
        if key_path.is_file():
            status = "DONE"
            completed_weight += weight
            mtime = datetime.fromtimestamp(key_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        else:
            status = "PENDING"
            mtime = "-"
            
        rows.append({
            "phase": num,
            "title": info["title"],
            "status": status,
            "key_file": info["key_file"],
            "date": mtime,
            "weight": weight
        })
        
    pct_complete = (completed_weight / total_weight * 100.0) if total_weight else 0.0
    return rows, pct_complete, completed_weight, total_weight

def print_status_table():
    rows, pct_complete, completed_weight, total_weight = get_status()
    print("\n" + "=" * 96)
    print("                     HOST-BASED INTRUSION DETECTION PROJECT STATUS")
    print("=" * 96)
    print(f"{'Phase':<7s} {'Title':<45s} {'Weight':<8s} {'Status':<10s} {'Last Modified':<18s}")
    print("-" * 96)
    for r in rows:
        st_color = r["status"]
        print(f"Phase {r['phase']:<1d} {r['title']:<45s} {r['weight']:>3d}%     {st_color:<10s} {r['date']:<18s}")
        print(f"        -> Output: {r['key_file']}")
    print("-" * 96)
    print(f"Overall Completion: {pct_complete:.1f}% ({completed_weight} / {total_weight} points)")
    print("=" * 96 + "\n")

if __name__ == "__main__":
    print_status_table()
