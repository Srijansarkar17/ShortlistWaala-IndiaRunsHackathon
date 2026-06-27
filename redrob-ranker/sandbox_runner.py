#!/usr/bin/env python3
"""
sandbox_runner.py — Orchestrates the full sandbox test inside Docker
====================================================================
This is the ENTRYPOINT for the validation Dockerfile.

It does NOT run or inspect the original ranking code. Instead it:
  1. Validates the submission CSV format against the spec
  2. Checks candidate_ids against the provided candidates.jsonl
  3. Validates repository structure
  4. Reports a comprehensive pass/fail summary

Usage inside Docker (via ENTRYPOINT):
  # Paths are mounted at Docker run time
  python sandbox_runner.py \
      --csv /workspace/submission.csv \
      --candidates /workspace/candidates.jsonl \
      --repo-root /workspace/repo
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ──────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(text: str):
    width = 70
    print()
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}")
    print()


def section(num: int, text: str):
    print(f"\n{BOLD}{CYAN}[Step {num}]{RESET} {BOLD}{text}{RESET}")
    print(f"{CYAN}{'─' * 50}{RESET}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Redrob Sandbox Runner")
    parser.add_argument("--csv", default="/workspace/submission.csv",
                        help="Path to submission CSV")
    parser.add_argument("--candidates", default="/workspace/candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--repo-root", default="/workspace/repo",
                        help="Path to repo root for structure check")
    parser.add_argument("--run-rank", action="store_true",
                        help="Also run rank.py inside sandbox (measures time/memory)")
    parser.add_argument("--rank-cmd", default=None,
                        help="Custom command to run the ranker (e.g. 'python rank.py --candidates ...')")
    args = parser.parse_args()

    start_time = time.time()

    banner("REDROB HACKATHON v4 — SANDBOX VALIDATION")

    print(f"  {CYAN}Timestamp:{RESET}    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {CYAN}CSV:{RESET}          {args.csv}")
    print(f"  {CYAN}Candidates:{RESET}   {args.candidates}")
    print(f"  {CYAN}Repo Root:{RESET}    {args.repo_root}")
    print()

    # ─── Step 1: Environment check ───────────────────────────────────────
    section(1, "Environment Check")
    
    print(f"  Python version: {sys.version.split()[0]}")
    
    # Check memory limit (if running in Docker with --memory)
    try:
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
            mem_limit = int(f.read().strip())
            mem_gb = mem_limit / (1024 ** 3)
            if mem_gb < 100:  # Real limit set
                print(f"  Memory limit: {mem_gb:.1f} GB")
                if mem_gb > 16:
                    print(f"  {YELLOW}⚠ Memory limit > 16 GB — actual sandbox is 16 GB{RESET}")
            else:
                print(f"  Memory limit: Not constrained (host default)")
    except (FileNotFoundError, PermissionError):
        try:
            # cgroup v2
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                val = f.read().strip()
                if val == "max":
                    print(f"  Memory limit: Not constrained (host default)")
                else:
                    mem_gb = int(val) / (1024 ** 3)
                    print(f"  Memory limit: {mem_gb:.1f} GB")
        except (FileNotFoundError, PermissionError):
            print(f"  Memory limit: Unknown (not in Docker cgroup?)")

    # Check network
    import socket
    try:
        socket.setdefaulttimeout(2)
        socket.create_connection(("8.8.8.8", 53))
        print(f"  {YELLOW}⚠ Network is AVAILABLE — actual sandbox blocks network (--network none){RESET}")
    except (socket.timeout, OSError):
        print(f"  {GREEN}✔ Network is blocked (as required){RESET}")

    # ─── Step 2: Optional rank.py execution ──────────────────────────────
    if args.run_rank:
        section(2, "Running Ranking Code (Optional)")
        
        rank_start = time.time()
        
        if args.rank_cmd:
            cmd = args.rank_cmd
        else:
            # Default command
            cmd = f"python /workspace/repo/rank.py --candidates {args.candidates} --out {args.csv}"
        
        print(f"  Command: {cmd}")
        print(f"  Timeout: 300 seconds (5 minutes)")
        print()
        
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                cwd="/workspace",
            )
            rank_elapsed = time.time() - rank_start
            
            if proc.returncode == 0:
                print(f"  {GREEN}✔ Ranking completed in {rank_elapsed:.1f}s{RESET}")
                if rank_elapsed > 300:
                    print(f"  {RED}✘ Exceeded 5-minute limit!{RESET}")
                elif rank_elapsed > 240:
                    print(f"  {YELLOW}⚠ Very close to 5-minute limit ({rank_elapsed:.1f}s){RESET}")
            else:
                print(f"  {RED}✘ Ranking failed (exit code {proc.returncode}){RESET}")
                if proc.stderr:
                    print(f"\n  STDERR (last 500 chars):")
                    print(f"  {proc.stderr[-500:]}")
                    
        except subprocess.TimeoutExpired:
            rank_elapsed = time.time() - rank_start
            print(f"  {RED}✘ TIMEOUT: Ranking exceeded 5-minute wall-clock limit ({rank_elapsed:.1f}s){RESET}")
            
    else:
        section(2, "Skipping Ranking Code Execution (use --run-rank to enable)")

    # ─── Step 3: CSV Validation ──────────────────────────────────────────
    section(3, "Submission CSV Validation")
    
    csv_path = Path(args.csv)
    candidates_path = Path(args.candidates) if args.candidates else None
    repo_root = Path(args.repo_root) if args.repo_root else None
    
    # Build validation command
    validate_cmd = [
        sys.executable, "/app/validate_submission.py",
        "--csv", str(csv_path),
    ]
    
    if candidates_path and candidates_path.exists():
        validate_cmd.extend(["--candidates", str(candidates_path)])
    
    if repo_root and repo_root.exists():
        validate_cmd.extend(["--repo-root", str(repo_root)])
    
    print(f"  Running: {' '.join(validate_cmd)}")
    print()
    
    proc = subprocess.run(validate_cmd, cwd="/app")
    
    # ─── Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    
    banner("SANDBOX VALIDATION COMPLETE")
    
    print(f"  Total time: {total_elapsed:.1f}s")
    
    if proc.returncode == 0:
        print(f"  {GREEN}{BOLD}RESULT: ALL CHECKS PASSED ✓{RESET}")
        print(f"\n  Your submission meets all format requirements.")
        print(f"  Remember: compute constraints are enforced during Stage 3 reproduction.")
    else:
        print(f"  {RED}{BOLD}RESULT: VALIDATION FAILED ✘{RESET}")
        print(f"\n  Please fix the issues above before submitting.")
    
    print()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
