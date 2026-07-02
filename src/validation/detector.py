"""
src/validation/detector.py
===========================
HoneypotDetector
----------------
Runs all deterministic checks against a `Candidate` and produces a
flat `DetectionResult` record with:

  - honeypot_score  : float in [0, 1]  — 0 = clean, 1 = definite honeypot
  - trust_score     : float in [0, 1]  — 1 - honeypot_score (monotone inverse)
  - is_honeypot     : bool             — True when honeypot_score > threshold
  - checks_triggered: list[str]        — names of fired checks
  - total_penalty   : float            — raw penalty sum before clipping
  - per_check       : dict             — detailed per-check breakdown

Design
------
  honeypot_score = clip(sum(penalty_i for triggered checks), 0, 1)
  trust_score    = 1 - honeypot_score

The threshold for is_honeypot defaults to 0.6 (configurable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.models import Candidate
from .checks import ALL_CHECKS, CheckResult, CheckName


_DEFAULT_THRESHOLD = 0.6


@dataclass
class DetectionResult:
    """Flat output record for one candidate's honeypot evaluation."""

    candidate_id: str

    # Primary scores
    honeypot_score: float   # [0, 1] — higher = more suspicious
    trust_score: float      # [0, 1] — 1 - honeypot_score

    # Decision
    is_honeypot: bool

    # Diagnostics
    checks_triggered: list[str]       # list of CheckName values that fired
    total_penalty: float              # raw penalty sum (may exceed 1 before clip)
    num_checks_run: int
    num_checks_triggered: int

    # Per-check details (serialisable as JSON for Parquet)
    check_reasons: dict[str, str]     # check_name → reason string
    check_penalties: dict[str, float] # check_name → penalty applied


class HoneypotDetector:
    """
    Runs the full deterministic check suite on a Candidate.

    Parameters
    ----------
    threshold : float
        honeypot_score cutoff above which is_honeypot=True. Default 0.6.
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    def detect(self, candidate: Candidate) -> DetectionResult:
        """Run all checks, return a DetectionResult."""
        results: list[CheckResult] = [check(candidate) for check in ALL_CHECKS]

        triggered = [r for r in results if r.triggered]
        raw_penalty = sum(r.penalty for r in triggered)
        honeypot_score = round(min(max(raw_penalty, 0.0), 1.0), 6)
        trust_score = round(1.0 - honeypot_score, 6)

        return DetectionResult(
            candidate_id=candidate.candidate_id,
            honeypot_score=honeypot_score,
            trust_score=trust_score,
            is_honeypot=honeypot_score > self.threshold,
            checks_triggered=[r.check.value for r in triggered],
            total_penalty=round(raw_penalty, 6),
            num_checks_run=len(results),
            num_checks_triggered=len(triggered),
            check_reasons={r.check.value: r.reason for r in results},
            check_penalties={r.check.value: r.penalty for r in triggered},
        )

    def detect_batch(self, candidates: list[Candidate]) -> list[DetectionResult]:
        """Run detection on a list of candidates."""
        return [self.detect(c) for c in candidates]
