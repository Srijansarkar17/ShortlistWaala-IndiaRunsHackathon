#!/usr/bin/env python3
"""
validate_submission.py — Comprehensive Redrob Hackathon v4 Submission Validator
================================================================================
Validates a submission CSV against ALL rules in the submission specification.
Also validates repository structure and metadata presence.

This script does NOT inspect the original ranking code content — it only checks
that the submission artefacts satisfy the spec.

Exit codes:
  0 — All checks passed
  1 — One or more checks failed

Usage:
  python validate_submission.py \
      --csv submission.csv \
      --candidates candidates.jsonl \
      [--repo-root /path/to/repo] \
      [--verbose]
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colours for terminal output
# ──────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS_ICON = f"{GREEN}✔ PASS{RESET}"
FAIL_ICON = f"{RED}✘ FAIL{RESET}"
WARN_ICON = f"{YELLOW}⚠ WARN{RESET}"
INFO_ICON = f"{CYAN}ℹ INFO{RESET}"

# ──────────────────────────────────────────────────────────────────────────────
# Constants from spec
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
EXPECTED_DATA_ROWS = 100
RANK_MIN = 1
RANK_MAX = 100


class ValidationResult:
    """Accumulates pass/fail/warn results across all checks."""

    def __init__(self):
        self.checks: list[tuple[str, str, str]] = []  # (status, category, message)
        self._pass_count = 0
        self._fail_count = 0
        self._warn_count = 0

    def passed(self, category: str, message: str):
        self.checks.append(("PASS", category, message))
        self._pass_count += 1

    def failed(self, category: str, message: str):
        self.checks.append(("FAIL", category, message))
        self._fail_count += 1

    def warned(self, category: str, message: str):
        self.checks.append(("WARN", category, message))
        self._warn_count += 1

    def info(self, category: str, message: str):
        self.checks.append(("INFO", category, message))

    @property
    def is_valid(self) -> bool:
        return self._fail_count == 0

    def summary(self) -> str:
        return (
            f"Passed: {self._pass_count}  |  "
            f"Failed: {self._fail_count}  |  "
            f"Warnings: {self._warn_count}"
        )

    def print_report(self):
        print()
        print(f"{BOLD}{'═' * 70}{RESET}")
        print(f"{BOLD}  REDROB HACKATHON v4 — SUBMISSION VALIDATION REPORT{RESET}")
        print(f"{BOLD}{'═' * 70}{RESET}")
        print()

        current_category = None
        for status, category, message in self.checks:
            if category != current_category:
                current_category = category
                print(f"\n{BOLD}┌─ {category}{RESET}")

            if status == "PASS":
                icon = PASS_ICON
            elif status == "FAIL":
                icon = FAIL_ICON
            elif status == "WARN":
                icon = WARN_ICON
            else:
                icon = INFO_ICON

            print(f"│  {icon}  {message}")

        print()
        print(f"{BOLD}{'─' * 70}{RESET}")
        overall = f"{GREEN}ALL CHECKS PASSED{RESET}" if self.is_valid else f"{RED}VALIDATION FAILED{RESET}"
        print(f"  {BOLD}{overall}{RESET}  —  {self.summary()}")
        print(f"{BOLD}{'─' * 70}{RESET}")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# 1. File-level checks
# ──────────────────────────────────────────────────────────────────────────────

def check_file_basics(csv_path: Path, result: ValidationResult):
    """Checks: file exists, .csv extension, UTF-8 encoding."""
    cat = "1. FILE FORMAT"

    if not csv_path.exists():
        result.failed(cat, f"File not found: {csv_path}")
        return False

    result.passed(cat, f"File exists: {csv_path.name}")

    # Extension check
    if csv_path.suffix.lower() != ".csv":
        result.failed(cat, f"Extension must be .csv, got '{csv_path.suffix}'")
    else:
        result.passed(cat, "Extension is .csv")

    # UTF-8 check
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            f.read()
        result.passed(cat, "File is valid UTF-8")
    except UnicodeDecodeError:
        result.failed(cat, "File is NOT valid UTF-8 encoding")
        return False

    # Check not xlsx/json/etc
    for bad_ext in [".xlsx", ".xls", ".json", ".jsonl", ".tsv", ".txt"]:
        if csv_path.suffix.lower() == bad_ext:
            result.failed(cat, f"Wrong format: submitted as {bad_ext}, must be .csv")

    return True


# ──────────────────────────────────────────────────────────────────────────────
# 2. Header & row-count checks
# ──────────────────────────────────────────────────────────────────────────────

def parse_csv(csv_path: Path, result: ValidationResult) -> Optional[list[dict]]:
    """Parse CSV, validate header and row count. Returns list of row dicts or None."""
    cat = "2. CSV STRUCTURE"

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        # Header
        try:
            header = next(reader)
        except StopIteration:
            result.failed(cat, "File is empty — no header row found")
            return None

        # Strip whitespace from header columns
        header = [h.strip() for h in header]

        if header == REQUIRED_HEADER:
            result.passed(cat, f"Header is correct: {','.join(header)}")
        else:
            result.failed(cat, f"Header must be {','.join(REQUIRED_HEADER)}, got: {','.join(header)}")
            return None

        # Read data rows, skip truly empty rows
        data_rows = []
        for row in reader:
            if any(cell.strip() for cell in row):
                data_rows.append(row)

    # Row count
    n = len(data_rows)
    if n == EXPECTED_DATA_ROWS:
        result.passed(cat, f"Exactly {EXPECTED_DATA_ROWS} data rows ✓")
    else:
        result.failed(cat, f"Expected {EXPECTED_DATA_ROWS} data rows, found {n}")
        if n == 99:
            result.info(cat, "Common mistake: 99 rows — did you accidentally skip one?")
        elif n == 101:
            result.info(cat, "Common mistake: 101 rows — did you include an extra row?")

    # Convert to dicts
    rows = []
    for i, cells in enumerate(data_rows):
        if len(cells) != len(REQUIRED_HEADER):
            result.failed(cat, f"Row {i+2}: expected {len(REQUIRED_HEADER)} columns, got {len(cells)}")
            continue
        rows.append(dict(zip(REQUIRED_HEADER, cells)))

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# 3. Column-level validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_columns(rows: list[dict], result: ValidationResult) -> tuple[list, list, list]:
    """Validate candidate_id, rank, score columns. Returns parsed lists."""
    cat = "3. COLUMN VALIDATION"

    candidate_ids = []
    ranks = []
    scores = []
    errors = 0

    for i, row in enumerate(rows):
        row_num = i + 2

        # candidate_id
        cid = row["candidate_id"].strip()
        if not cid:
            result.failed(cat, f"Row {row_num}: candidate_id is empty")
            errors += 1
        elif not CANDIDATE_ID_PATTERN.match(cid):
            result.failed(cat, f"Row {row_num}: candidate_id '{cid}' doesn't match CAND_XXXXXXX format")
            errors += 1
        candidate_ids.append(cid)

        # rank
        rank_s = row["rank"].strip()
        try:
            rank = int(rank_s)
            if str(rank) != rank_s:
                raise ValueError
            if not (RANK_MIN <= rank <= RANK_MAX):
                result.failed(cat, f"Row {row_num}: rank {rank} is outside 1-100 range")
                errors += 1
            ranks.append(rank)
        except ValueError:
            result.failed(cat, f"Row {row_num}: rank '{rank_s}' is not a valid integer")
            ranks.append(None)
            errors += 1

        # score
        score_s = row["score"].strip()
        try:
            score = float(score_s)
            if math.isnan(score) or math.isinf(score):
                raise ValueError
            scores.append(score)
        except ValueError:
            result.failed(cat, f"Row {row_num}: score '{score_s}' is not a valid float")
            scores.append(None)
            errors += 1

    if errors == 0:
        result.passed(cat, "All candidate_id values match CAND_XXXXXXX format")
        result.passed(cat, "All rank values are valid integers")
        result.passed(cat, "All score values are valid floats")

    return candidate_ids, ranks, scores


# ──────────────────────────────────────────────────────────────────────────────
# 4. Uniqueness & completeness checks
# ──────────────────────────────────────────────────────────────────────────────

def validate_uniqueness(candidate_ids: list, ranks: list, result: ValidationResult):
    """Check for duplicates in candidate_id and rank; check rank completeness."""
    cat = "4. UNIQUENESS & COMPLETENESS"

    # Duplicate candidate_ids
    cid_counts = Counter(candidate_ids)
    dups = {k: v for k, v in cid_counts.items() if v > 1}
    if dups:
        for cid, count in dups.items():
            result.failed(cat, f"Duplicate candidate_id '{cid}' appears {count} times")
    else:
        result.passed(cat, "All candidate_ids are unique")

    # Duplicate ranks
    valid_ranks = [r for r in ranks if r is not None]
    rank_counts = Counter(valid_ranks)
    dup_ranks = {k: v for k, v in rank_counts.items() if v > 1}
    if dup_ranks:
        for rank, count in dup_ranks.items():
            result.failed(cat, f"Duplicate rank {rank} appears {count} times")
    else:
        result.passed(cat, "All ranks are unique")

    # Ranks must cover 1..100 completely
    expected = set(range(1, 101))
    actual = set(valid_ranks)
    missing = expected - actual
    extra = actual - expected

    if not missing and not extra:
        result.passed(cat, "Ranks 1-100 are all present exactly once")
    else:
        if missing:
            result.failed(cat, f"Missing ranks: {sorted(missing)}")
        if extra:
            result.failed(cat, f"Extra ranks outside 1-100: {sorted(extra)}")

    # Common mistake: ranks starting at 0
    if 0 in actual:
        result.failed(cat, "Rank starts at 0 — ranks must start at 1")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Score monotonicity check
# ──────────────────────────────────────────────────────────────────────────────

def validate_score_ordering(rows: list[dict], ranks: list, scores: list, result: ValidationResult):
    """Score must be non-increasing as rank increases."""
    cat = "5. SCORE ORDERING"

    # Build (rank, score, cid) triples for valid entries
    triples = []
    for i in range(len(rows)):
        if ranks[i] is not None and scores[i] is not None:
            triples.append((ranks[i], scores[i], rows[i]["candidate_id"].strip()))

    triples.sort(key=lambda x: x[0])

    # Check non-increasing scores
    violations = 0
    for i in range(len(triples) - 1):
        r1, s1, c1 = triples[i]
        r2, s2, c2 = triples[i + 1]
        if s1 < s2:
            violations += 1
            if violations <= 3:
                result.failed(cat, f"Score increases from rank {r1} ({s1}) to rank {r2} ({s2})")

    if violations > 3:
        result.failed(cat, f"... and {violations - 3} more score ordering violations")

    if violations == 0:
        result.passed(cat, "Scores are non-increasing with rank ✓")

    # Check all scores identical (warning)
    unique_scores = set(s for s in scores if s is not None)
    if len(unique_scores) == 1:
        result.failed(cat, f"All scores are identical ({unique_scores.pop()}) — model is not differentiating")
    elif len(unique_scores) <= 3:
        result.warned(cat, f"Only {len(unique_scores)} distinct score values — very low differentiation")
    else:
        result.passed(cat, f"{len(unique_scores)} distinct score values — good differentiation")

    # Check tie-breaking: equal scores should have candidate_id ascending
    tie_violations = 0
    for i in range(len(triples) - 1):
        r1, s1, c1 = triples[i]
        r2, s2, c2 = triples[i + 1]
        if s1 == s2 and c1 > c2:
            tie_violations += 1
            if tie_violations <= 2:
                result.warned(cat, f"Score tie at ranks {r1},{r2}: candidate_id should be ascending ({c1} > {c2})")

    if tie_violations == 0:
        result.passed(cat, "Score ties broken correctly by candidate_id ascending")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Candidate existence check against candidates.jsonl
# ──────────────────────────────────────────────────────────────────────────────

def validate_candidate_existence(candidate_ids: list, candidates_path: Optional[Path], result: ValidationResult):
    """Verify every candidate_id exists in the released candidates.jsonl."""
    cat = "6. CANDIDATE EXISTENCE"

    if candidates_path is None or not candidates_path.exists():
        result.warned(cat, "candidates.jsonl not provided — skipping candidate existence check")
        return

    result.info(cat, f"Loading candidate IDs from {candidates_path.name} (may take a moment)...")

    valid_ids = set()
    line_count = 0
    try:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj.get("candidate_id", "")
                    if cid:
                        valid_ids.add(cid)
                    line_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        result.warned(cat, f"Error reading candidates file: {e}")
        return

    result.info(cat, f"Loaded {len(valid_ids):,} candidate IDs from {line_count:,} lines")

    # Check each submission candidate_id exists
    missing = []
    for cid in candidate_ids:
        if cid and cid not in valid_ids:
            missing.append(cid)

    if not missing:
        result.passed(cat, "All 100 candidate_ids exist in candidates.jsonl ✓")
    else:
        for cid in missing[:5]:
            result.failed(cat, f"candidate_id '{cid}' does NOT exist in candidates.jsonl")
        if len(missing) > 5:
            result.failed(cat, f"... and {len(missing) - 5} more missing candidate_ids")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Reasoning quality checks (warnings, not failures)
# ──────────────────────────────────────────────────────────────────────────────

def validate_reasoning(rows: list[dict], result: ValidationResult):
    """Check reasoning column quality — warnings, not hard failures."""
    cat = "7. REASONING QUALITY"

    reasonings = [row.get("reasoning", "").strip() for row in rows]

    # Check if all empty
    non_empty = [r for r in reasonings if r]
    if not non_empty:
        result.warned(cat, "All reasoning fields are empty — strongly recommended to include reasoning")
        return

    if len(non_empty) < 100:
        result.warned(cat, f"Only {len(non_empty)}/100 rows have reasoning — recommended for all rows")
    else:
        result.passed(cat, "All 100 rows have reasoning text")

    # Check if all identical
    unique_reasonings = set(non_empty)
    if len(unique_reasonings) == 1:
        result.warned(cat, "All reasoning strings are IDENTICAL — will be penalized at Stage 4")
    elif len(unique_reasonings) < 10:
        result.warned(cat, f"Only {len(unique_reasonings)} unique reasoning strings — likely templated")
    else:
        result.passed(cat, f"{len(unique_reasonings)} unique reasoning strings — good variation")

    # Check average length
    avg_len = sum(len(r) for r in non_empty) / len(non_empty)
    if avg_len < 20:
        result.warned(cat, f"Average reasoning length is {avg_len:.0f} chars — too short for meaningful reasoning")
    elif avg_len < 50:
        result.warned(cat, f"Average reasoning length is {avg_len:.0f} chars — consider adding more detail")
    else:
        result.passed(cat, f"Average reasoning length: {avg_len:.0f} chars")

    # Check for very short reasonings
    short = [r for r in non_empty if len(r) < 15]
    if short:
        result.warned(cat, f"{len(short)} reasoning entries are under 15 characters")

    # Rank consistency: basic check for top-10 vs bottom-10 tone
    # (This is a heuristic — real review is manual)
    top_10_reasonings = []
    bottom_10_reasonings = []
    for row in rows:
        rank_s = row["rank"].strip()
        try:
            rank = int(rank_s)
            reasoning = row.get("reasoning", "").strip()
            if reasoning:
                if rank <= 10:
                    top_10_reasonings.append(reasoning)
                elif rank >= 91:
                    bottom_10_reasonings.append(reasoning)
        except ValueError:
            continue

    if top_10_reasonings and bottom_10_reasonings:
        # Just check that top-10 and bottom-10 aren't identical
        if set(top_10_reasonings) == set(bottom_10_reasonings):
            result.warned(cat, "Top-10 and bottom-10 reasonings are identical — rank consistency concern")
        else:
            result.passed(cat, "Top-10 and bottom-10 reasonings differ (rank consistency OK)")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Repository structure check
# ──────────────────────────────────────────────────────────────────────────────

def validate_repo_structure(repo_root: Optional[Path], result: ValidationResult):
    """Check repository has required files per Section 10.3."""
    cat = "8. REPOSITORY STRUCTURE"

    if repo_root is None or not repo_root.exists():
        result.warned(cat, "Repository root not provided — skipping repo structure check")
        return

    result.info(cat, f"Checking repo at: {repo_root}")

    # README.md
    readme = repo_root / "README.md"
    if readme.exists():
        result.passed(cat, "README.md exists")
        # Check it has content
        content = readme.read_text(encoding="utf-8", errors="replace")
        if len(content) < 50:
            result.warned(cat, "README.md is very short — ensure it has setup instructions")
        # Check for reproduction command
        if "rank.py" in content or "python" in content.lower():
            result.passed(cat, "README.md appears to reference a run command")
        else:
            result.warned(cat, "README.md should include exact commands to reproduce submission CSV")
    else:
        result.failed(cat, "README.md is MISSING — required for submission")

    # requirements.txt or pyproject.toml
    has_deps = (
        (repo_root / "requirements.txt").exists()
        or (repo_root / "pyproject.toml").exists()
        or (repo_root / "setup.py").exists()
        or (repo_root / "setup.cfg").exists()
    )
    if has_deps:
        result.passed(cat, "Dependency file found (requirements.txt / pyproject.toml)")
    else:
        result.failed(cat, "No dependency file found — need requirements.txt or pyproject.toml")

    # submission_metadata.yaml
    meta_yaml = repo_root / "submission_metadata.yaml"
    if not meta_yaml.exists():
        meta_yaml = repo_root / "submission_metadata.yml"
    if meta_yaml.exists():
        result.passed(cat, f"{meta_yaml.name} exists")
    else:
        result.warned(cat, "submission_metadata.yaml not found — required at repo root")

    # Source code (at least one .py file)
    py_files = list(repo_root.rglob("*.py"))
    # Exclude __pycache__ and .git
    py_files = [f for f in py_files if "__pycache__" not in str(f) and ".git" not in str(f)]
    if py_files:
        result.passed(cat, f"Found {len(py_files)} Python source files")
    else:
        result.failed(cat, "No Python source files found in repo")

    # Dockerfile or docker setup
    if (repo_root / "Dockerfile").exists():
        result.passed(cat, "Dockerfile exists")
    else:
        result.warned(cat, "No Dockerfile found — recommended for sandbox testing")

    # .gitignore
    if (repo_root / ".gitignore").exists():
        result.passed(cat, ".gitignore exists")
    else:
        result.warned(cat, "No .gitignore — recommended to exclude large data files")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Compute constraints summary (info only — checked at runtime)
# ──────────────────────────────────────────────────────────────────────────────

def print_compute_constraints(result: ValidationResult):
    """Remind about compute constraints — these are enforced at Docker runtime."""
    cat = "9. COMPUTE CONSTRAINTS (REMINDER)"

    result.info(cat, "Runtime limit: ≤ 5 minutes wall-clock")
    result.info(cat, "Memory limit: ≤ 16 GB RAM")
    result.info(cat, "Compute: CPU only — no GPU")
    result.info(cat, "Network: OFF — no external API calls")
    result.info(cat, "Disk: ≤ 5 GB intermediate state")
    result.info(cat, "These are enforced by the Docker sandbox at runtime")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Redrob Hackathon v4 — Comprehensive Submission Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate CSV only
  python validate_submission.py --csv submission.csv

  # Validate CSV + check candidate_ids against dataset
  python validate_submission.py --csv submission.csv --candidates candidates.jsonl

  # Full validation including repo structure
  python validate_submission.py --csv submission.csv --candidates candidates.jsonl --repo-root .
        """,
    )
    parser.add_argument("--csv", required=True, help="Path to submission CSV file")
    parser.add_argument("--candidates", default=None, help="Path to candidates.jsonl for ID validation")
    parser.add_argument("--repo-root", default=None, help="Path to repo root for structure validation")
    parser.add_argument("--verbose", action="store_true", help="Show extra debug info")

    args = parser.parse_args()
    result = ValidationResult()

    csv_path = Path(args.csv)
    candidates_path = Path(args.candidates) if args.candidates else None
    repo_root = Path(args.repo_root) if args.repo_root else None

    # ── 1. File basics ──
    if not check_file_basics(csv_path, result):
        result.print_report()
        sys.exit(1)

    # ── 2. Parse CSV ──
    rows = parse_csv(csv_path, result)
    if rows is None:
        result.print_report()
        sys.exit(1)

    # ── 3. Column validation ──
    candidate_ids, ranks, scores = validate_columns(rows, result)

    # ── 4. Uniqueness & completeness ──
    validate_uniqueness(candidate_ids, ranks, result)

    # ── 5. Score ordering ──
    validate_score_ordering(rows, ranks, scores, result)

    # ── 6. Candidate existence ──
    validate_candidate_existence(candidate_ids, candidates_path, result)

    # ── 7. Reasoning quality ──
    validate_reasoning(rows, result)

    # ── 8. Repository structure ──
    validate_repo_structure(repo_root, result)

    # ── 9. Compute constraints reminder ──
    print_compute_constraints(result)

    # ── Print report ──
    result.print_report()

    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
