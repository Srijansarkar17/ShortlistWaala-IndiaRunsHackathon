"""
tests/test_checks.py
====================
Unit tests for all 7 deterministic honeypot checks.

Each check is tested with:
  - A clean candidate (should NOT trigger)
  - A manipulated candidate (SHOULD trigger)
  - Edge cases (None fields, boundary conditions)
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.ingestion.models import (
    Candidate,
    CandidateProfile,
    CareerEntry,
    Certification,
    Education,
    RedrobSignals,
    Skill,
)
from src.validation.checks import (
    CheckName,
    check_duration_mismatch,
    check_education_date_inversion,
    check_expert_zero_duration,
    check_job_date_inversion,
    check_salary_inversion,
    check_skill_exceeds_experience,
    check_startup_founding_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(
    candidate_id: str = "CAND_TEST",
    yoe: float | None = 5.0,
    salary: dict | None = None,
    career: list[CareerEntry] | None = None,
    education: list[Education] | None = None,
    skills: list[Skill] | None = None,
) -> Candidate:
    signals = RedrobSignals(expected_salary_range_inr_lpa=salary)
    return Candidate(
        candidate_id=candidate_id,
        profile=CandidateProfile(years_of_experience=yoe),
        career_history=career or [],
        education=education or [],
        skills=skills or [],
        redrob_signals=signals,
    )


def _job(
    company: str = "Acme",
    title: str = "Engineer",
    start: str | None = "2020-01-01",
    end: str | None = "2022-01-01",
    duration: int | None = 24,
    is_current: bool = False,
) -> CareerEntry:
    return CareerEntry(
        company=company,
        title=title,
        start_date=start,
        end_date=end,
        duration_months=duration,
        is_current=is_current,
    )


def _skill(name: str, proficiency: str = "intermediate", duration: int | None = 12) -> Skill:
    return Skill(name=name, proficiency=proficiency, duration_months=duration)


# ---------------------------------------------------------------------------
# C1 — Salary Inversion
# ---------------------------------------------------------------------------


class TestSalaryInversion:
    def test_no_salary_clean(self):
        r = check_salary_inversion(_candidate(salary=None))
        assert not r.triggered

    def test_valid_range_clean(self):
        r = check_salary_inversion(_candidate(salary={"min": 10.0, "max": 20.0}))
        assert not r.triggered

    def test_equal_min_max_clean(self):
        r = check_salary_inversion(_candidate(salary={"min": 15.0, "max": 15.0}))
        assert not r.triggered

    def test_inverted_salary_triggers(self):
        r = check_salary_inversion(_candidate(salary={"min": 50.0, "max": 20.0}))
        assert r.triggered
        assert r.check == CheckName.SALARY_INVERSION
        assert r.penalty > 0

    def test_missing_min_field_clean(self):
        r = check_salary_inversion(_candidate(salary={"max": 20.0}))
        assert not r.triggered

    def test_missing_max_field_clean(self):
        r = check_salary_inversion(_candidate(salary={"min": 10.0}))
        assert not r.triggered

    def test_non_numeric_salary_clean(self):
        r = check_salary_inversion(_candidate(salary={"min": "abc", "max": "xyz"}))
        assert not r.triggered


# ---------------------------------------------------------------------------
# C2 — Job Date Inversion
# ---------------------------------------------------------------------------


class TestJobDateInversion:
    def test_valid_dates_clean(self):
        r = check_job_date_inversion(_candidate(career=[_job(start="2020-01-01", end="2022-01-01")]))
        assert not r.triggered

    def test_inverted_dates_triggers(self):
        r = check_job_date_inversion(_candidate(career=[_job(start="2022-01-01", end="2020-01-01")]))
        assert r.triggered
        assert r.penalty > 0

    def test_current_job_skipped(self):
        # even with impossible dates, current jobs are skipped
        r = check_job_date_inversion(_candidate(
            career=[CareerEntry(company="X", title="Y", start_date="2099-01-01", end_date=None, is_current=True)]
        ))
        assert not r.triggered

    def test_multiple_violations_accumulate(self):
        r = check_job_date_inversion(_candidate(career=[
            _job(start="2022-01-01", end="2020-01-01"),
            _job(start="2023-01-01", end="2021-01-01"),
        ]))
        assert r.triggered
        assert r.penalty > 0.25   # at least 2 × base penalty

    def test_no_dates_clean(self):
        r = check_job_date_inversion(_candidate(career=[_job(start=None, end=None)]))
        assert not r.triggered

    def test_penalty_capped_at_0_5(self):
        # 4 violations × 0.25 = 1.0, should cap at 0.50
        jobs = [_job(start="2022-01-01", end="2020-01-01") for _ in range(4)]
        r = check_job_date_inversion(_candidate(career=jobs))
        assert r.penalty <= 0.50


# ---------------------------------------------------------------------------
# C3 — Education Date Inversion
# ---------------------------------------------------------------------------


class TestEducationDateInversion:
    def test_valid_edu_clean(self):
        r = check_education_date_inversion(_candidate(education=[
            Education(institution="MIT", start_year=2015, end_year=2019)
        ]))
        assert not r.triggered

    def test_inverted_edu_triggers(self):
        r = check_education_date_inversion(_candidate(education=[
            Education(institution="MIT", start_year=2019, end_year=2015)
        ]))
        assert r.triggered
        assert r.penalty > 0

    def test_missing_years_clean(self):
        r = check_education_date_inversion(_candidate(education=[
            Education(institution="MIT", start_year=None, end_year=None)
        ]))
        assert not r.triggered

    def test_penalty_capped_at_0_3(self):
        edus = [Education(institution=f"Uni{i}", start_year=2019, end_year=2015) for i in range(5)]
        r = check_education_date_inversion(_candidate(education=edus))
        assert r.penalty <= 0.30


# ---------------------------------------------------------------------------
# C4 — Duration Mismatch
# ---------------------------------------------------------------------------


class TestDurationMismatch:
    def test_matching_duration_clean(self):
        # 2020-01-01 to 2022-01-01 = 24 months
        r = check_duration_mismatch(_candidate(career=[_job(
            start="2020-01-01", end="2022-01-01", duration=24
        )]))
        assert not r.triggered

    def test_within_tolerance_clean(self):
        # 24 months computed, declared 26 — diff=2 ≤ 3
        r = check_duration_mismatch(_candidate(career=[_job(
            start="2020-01-01", end="2022-01-01", duration=26
        )]))
        assert not r.triggered

    def test_exceeds_tolerance_triggers(self):
        # 24 months computed, declared 36 — diff=12 > 3
        r = check_duration_mismatch(_candidate(career=[_job(
            start="2020-01-01", end="2022-01-01", duration=36
        )]))
        assert r.triggered
        assert r.penalty > 0

    def test_no_duration_skipped(self):
        r = check_duration_mismatch(_candidate(career=[_job(duration=None)]))
        assert not r.triggered

    def test_no_dates_skipped(self):
        r = check_duration_mismatch(_candidate(career=[_job(start=None, end=None, duration=24)]))
        assert not r.triggered

    def test_penalty_capped_at_0_4(self):
        jobs = [_job(start="2020-01-01", end="2022-01-01", duration=60) for _ in range(6)]
        r = check_duration_mismatch(_candidate(career=jobs))
        assert r.penalty <= 0.40


# ---------------------------------------------------------------------------
# C5 — Expert Zero Duration
# ---------------------------------------------------------------------------


class TestExpertZeroDuration:
    def test_below_threshold_clean(self):
        skills = [_skill("Python", "advanced", duration=0),
                  _skill("Go", "expert", duration=0)]
        r = check_expert_zero_duration(_candidate(skills=skills))
        assert not r.triggered   # only 2, threshold is 3

    def test_at_threshold_triggers(self):
        skills = [_skill(f"Skill{i}", "advanced", duration=0) for i in range(3)]
        r = check_expert_zero_duration(_candidate(skills=skills))
        assert r.triggered
        assert r.penalty > 0

    def test_non_expert_ignored(self):
        skills = [_skill(f"Skill{i}", "intermediate", duration=0) for i in range(5)]
        r = check_expert_zero_duration(_candidate(skills=skills))
        assert not r.triggered

    def test_non_zero_expert_ignored(self):
        skills = [_skill(f"Skill{i}", "advanced", duration=12) for i in range(5)]
        r = check_expert_zero_duration(_candidate(skills=skills))
        assert not r.triggered

    def test_mixed_triggers_only_on_zero_experts(self):
        skills = [
            _skill("A", "advanced", 0),
            _skill("B", "expert", 0),
            _skill("C", "advanced", 0),   # 3 zeros — should trigger
            _skill("D", "advanced", 12),  # non-zero, ignored
        ]
        r = check_expert_zero_duration(_candidate(skills=skills))
        assert r.triggered


# ---------------------------------------------------------------------------
# C6 — Skill Exceeds Experience
# ---------------------------------------------------------------------------


class TestSkillExceedsExperience:
    def test_no_yoe_skipped(self):
        skills = [_skill("Python", duration=999)]
        r = check_skill_exceeds_experience(_candidate(yoe=None, skills=skills))
        assert not r.triggered

    def test_within_ceiling_clean(self):
        # yoe=5 → 60m + 12 = 72m ceiling
        skills = [_skill("Python", duration=72)]
        r = check_skill_exceeds_experience(_candidate(yoe=5.0, skills=skills))
        assert not r.triggered

    def test_exceeds_ceiling_triggers(self):
        # yoe=5 → ceiling=72m, skill has 100m
        skills = [_skill("Python", duration=100)]
        r = check_skill_exceeds_experience(_candidate(yoe=5.0, skills=skills))
        assert r.triggered
        assert r.penalty > 0

    def test_none_duration_skipped(self):
        skills = [_skill("Python", duration=None)]
        r = check_skill_exceeds_experience(_candidate(yoe=5.0, skills=skills))
        assert not r.triggered

    def test_penalty_capped_at_0_4(self):
        skills = [_skill(f"S{i}", duration=999) for i in range(5)]
        r = check_skill_exceeds_experience(_candidate(yoe=1.0, skills=skills))
        assert r.penalty <= 0.40


# ---------------------------------------------------------------------------
# C7 — Startup Founding Year
# ---------------------------------------------------------------------------


class TestStartupFoundingYear:
    def test_no_startup_clean(self):
        career = [_job(company="Acme Corp", start="2018-01-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert not r.triggered

    def test_after_founding_clean(self):
        # OpenAI founded 2015, start 2016 → ok
        career = [_job(company="OpenAI", start="2016-06-01", end="2019-01-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert not r.triggered

    def test_before_founding_triggers(self):
        # OpenAI founded 2015, claiming to work there in 2013
        career = [_job(company="OpenAI", start="2013-01-01", end="2014-12-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert r.triggered
        assert r.penalty > 0

    def test_anthropic_2021(self):
        career = [_job(company="Anthropic", start="2019-01-01", end="2020-01-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert r.triggered

    def test_mistral_2023(self):
        career = [_job(company="Mistral AI", start="2022-01-01", end="2022-12-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert r.triggered

    def test_langchain_exact_founding_year_ok(self):
        # LangChain founded 2022 — starting in 2022 should be ok
        career = [_job(company="LangChain", start="2022-03-01", end="2023-01-01")]
        r = check_startup_founding_year(_candidate(career=career))
        assert not r.triggered

    def test_no_start_date_skipped(self):
        career = [CareerEntry(company="OpenAI", title="Eng", start_date=None)]
        r = check_startup_founding_year(_candidate(career=career))
        assert not r.triggered

    def test_penalty_capped_at_0_7(self):
        startups = ["OpenAI", "Anthropic", "Mistral AI"]
        jobs = [_job(company=s, start="2010-01-01", end="2011-01-01") for s in startups]
        r = check_startup_founding_year(_candidate(career=jobs))
        assert r.penalty <= 0.70
