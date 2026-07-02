"""
tests/test_detector.py
=======================
Unit tests for HoneypotDetector.

Tests:
- Clean candidate has honeypot_score == 0 and trust_score == 1
- Multiple triggered checks accumulate correctly
- score is clipped to [0, 1]
- is_honeypot flag respects threshold
- Custom threshold works
- detect_batch processes all candidates
- DetectionResult fields populated correctly
"""

from __future__ import annotations

import pytest

from src.ingestion.models import (
    Candidate,
    CandidateProfile,
    CareerEntry,
    Education,
    RedrobSignals,
    Skill,
)
from src.validation.detector import HoneypotDetector, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_candidate(cid: str = "CAND_CLEAN") -> Candidate:
    return Candidate(
        candidate_id=cid,
        profile=CandidateProfile(years_of_experience=5.0),
        redrob_signals=RedrobSignals(expected_salary_range_inr_lpa={"min": 10.0, "max": 30.0}),
    )


def _honeypot_candidate(cid: str = "CAND_HP") -> Candidate:
    """Candidate with multiple violations across checks."""
    return Candidate(
        candidate_id=cid,
        profile=CandidateProfile(years_of_experience=2.0),
        career_history=[
            # C2: start > end
            CareerEntry(
                company="Acme", title="Eng",
                start_date="2023-01-01", end_date="2021-01-01",
                duration_months=24, is_current=False,
            ),
            # C7: worked at OpenAI before it existed
            CareerEntry(
                company="OpenAI", title="Researcher",
                start_date="2012-01-01", end_date="2014-01-01",
                duration_months=24, is_current=False,
            ),
        ],
        skills=[
            # C5: 3 advanced skills with duration=0
            Skill(name="A", proficiency="advanced", duration_months=0),
            Skill(name="B", proficiency="advanced", duration_months=0),
            Skill(name="C", proficiency="advanced", duration_months=0),
        ],
        redrob_signals=RedrobSignals(
            # C1: salary inversion
            expected_salary_range_inr_lpa={"min": 100.0, "max": 10.0}
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHoneypotDetector:
    def setup_method(self):
        self.detector = HoneypotDetector(threshold=0.6)

    def test_clean_candidate_zero_score(self):
        result = self.detector.detect(_clean_candidate())
        assert result.honeypot_score == 0.0
        assert result.trust_score == 1.0
        assert not result.is_honeypot
        assert result.num_checks_triggered == 0

    def test_honeypot_candidate_triggers(self):
        result = self.detector.detect(_honeypot_candidate())
        assert result.honeypot_score > 0.0
        assert result.trust_score < 1.0
        assert result.num_checks_triggered >= 3

    def test_score_clipped_to_1(self):
        """When many checks trigger, score should not exceed 1.0."""
        result = self.detector.detect(_honeypot_candidate())
        assert result.honeypot_score <= 1.0
        assert result.trust_score >= 0.0

    def test_trust_plus_honeypot_equals_one(self):
        result = self.detector.detect(_honeypot_candidate())
        assert abs(result.honeypot_score + result.trust_score - 1.0) < 1e-9

    def test_is_honeypot_respects_threshold(self):
        # With threshold=1.0, nothing should be flagged
        lenient_detector = HoneypotDetector(threshold=1.0)
        result = lenient_detector.detect(_honeypot_candidate())
        assert not result.is_honeypot

    def test_is_honeypot_strict_threshold(self):
        # With threshold=0.0, everything gets flagged
        strict_detector = HoneypotDetector(threshold=0.0)
        # Even a "clean" candidate with any penalty gets flagged
        result = strict_detector.detect(_honeypot_candidate())
        assert result.is_honeypot

    def test_num_checks_run_equals_total_checks(self):
        result = self.detector.detect(_clean_candidate())
        assert result.num_checks_run == 7   # one per check

    def test_checks_triggered_list_non_empty_for_honeypot(self):
        result = self.detector.detect(_honeypot_candidate())
        assert len(result.checks_triggered) >= 1
        assert all(isinstance(c, str) for c in result.checks_triggered)

    def test_check_reasons_has_entry_for_every_check(self):
        result = self.detector.detect(_clean_candidate())
        assert len(result.check_reasons) == 7

    def test_check_penalties_only_for_triggered(self):
        result = self.detector.detect(_honeypot_candidate())
        assert set(result.check_penalties.keys()) == set(result.checks_triggered)

    def test_candidate_id_preserved(self):
        result = self.detector.detect(_clean_candidate("CAND_MYID"))
        assert result.candidate_id == "CAND_MYID"

    def test_detect_batch(self):
        candidates = [
            _clean_candidate(f"CAND_{i:03d}") for i in range(5)
        ] + [_honeypot_candidate()]
        results = self.detector.detect_batch(candidates)
        assert len(results) == 6
        scores = [r.honeypot_score for r in results]
        assert max(scores) > 0.0


class TestDetectionResultIntegrity:
    def test_total_penalty_equals_sum_of_check_penalties(self):
        detector = HoneypotDetector()
        result = detector.detect(_honeypot_candidate())
        expected = sum(result.check_penalties.values())
        assert abs(result.total_penalty - expected) < 1e-9

    def test_score_is_clipped_total_penalty(self):
        detector = HoneypotDetector()
        result = detector.detect(_honeypot_candidate())
        expected_score = min(result.total_penalty, 1.0)
        assert abs(result.honeypot_score - expected_score) < 1e-6
