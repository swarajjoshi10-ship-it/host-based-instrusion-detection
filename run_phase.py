"""
Runner script for individual HIDS phases.

Usage:
  python run_phase.py N          # Runs Phase N (e.g. python run_phase.py 1)
  python run_phase.py --upto N   # Runs Phases 1 through N sequentially
  python run_phase.py N --demo   # Runs Phase N in quick demo mode (< 1 min)
"""
import sys, os, subprocess, argparse
from pathlib import Path
from config import PHASE_REGISTRY, PROJECT_ROOT

def run_single_phase(num, demo=False):
    if num not in PHASE_REGISTRY:
        print(f"[ERROR] Phase {num} is not recognized. Available phases: 1 to {len(PHASE_REGISTRY)}.")
        return False
        
    info = PHASE_REGISTRY[num]
    script_path = PROJECT_ROOT / "phases" / info["script"]
    if not script_path.is_file():
        print(f"[ERROR] Script for Phase {num} not found at {script_path}")
        return False
        
    cmd = [sys.executable, str(script_path)]
    if demo:
        cmd.append("--demo")
        
    print(f"\n>>> Launching Phase {num}: {info['title']} (script: {info['script']}) {'[DEMO MODE]' if demo else ''}")
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        print(f"\n[FAILED] Phase {num} exited with error code {ret.returncode}.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="HIDS Phase Runner")
    parser.add_argument("phase", nargs="?", type=int, help="Phase number to run (1..8)")
    parser.add_argument("--upto", type=int, help="Run all phases from 1 up to N")
    parser.add_argument("--demo", action="store_true", help="Run with demo settings (single seed, faster execution)")
    
    args = parser.parse_args()
    
    if args.upto:
        target = args.upto
        print(f"Executing Phases 1 through {target} sequentially...\n")
        for p in range(1, target + 1):
            success = run_single_phase(p, demo=args.demo)
            if not success:
                print(f"\nExecution stopped at Phase {p} due to an error.")
                sys.exit(1)
        print(f"\nAll phases 1 through {target} finished successfully!")
    elif args.phase is not None:
        success = run_single_phase(args.phase, demo=args.demo)
        if not success:
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
