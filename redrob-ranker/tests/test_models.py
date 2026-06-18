"""
tests/test_models.py
====================
Unit tests for Pydantic candidate models.

Covers:
- Happy path construction
- Missing optional fields default correctly
- Malformed / null dates
- Empty / null skills filtered out
- Negative years_of_experience coerced to None
- ValidationError on missing candidate_id
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.ingestion.models import (
    Candidate,
    CandidateProfile,
    CareerEntry,
    Certification,
    Education,
    Language,
    RedrobSignals,
    Skill,
)


# ---------------------------------------------------------------------------
# CandidateProfile
# ---------------------------------------------------------------------------


class TestCandidateProfile:
    def test_happy_path(self):
        p = CandidateProfile(
            anonymized_name="Alice",
            headline="ML Engineer",
            summary="10 years in ML",
            years_of_experience=10.0,
        )
        assert p.headline == "ML Engineer"
        assert p.years_of_experience == 10.0

    def test_missing_optional_fields_default_none(self):
        p = CandidateProfile()
        assert p.location is None
        assert p.country is None
        assert p.current_title is None

    def test_none_headline_becomes_empty_string(self):
        p = CandidateProfile(headline=None)
        assert p.headline == ""

    def test_negative_yoe_becomes_none(self):
        p = CandidateProfile(years_of_experience=-5)
        assert p.years_of_experience is None

    def test_string_yoe_coerced(self):
        p = CandidateProfile(years_of_experience="7.5")
        assert p.years_of_experience == 7.5

    def test_nonsense_yoe_becomes_none(self):
        p = CandidateProfile(years_of_experience="abc")
        assert p.years_of_experience is None

    def test_whitespace_stripped_from_strings(self):
        p = CandidateProfile(headline="  Lead Engineer  ")
        assert p.headline == "Lead Engineer"


# ---------------------------------------------------------------------------
# CareerEntry
# ---------------------------------------------------------------------------


class TestCareerEntry:
    def test_valid_dates_parsed(self):
        e = CareerEntry(start_date="2020-01-15", end_date="2023-06-30")
        from datetime import date
        assert e.start_date == date(2020, 1, 15)
        assert e.end_date == date(2023, 6, 30)

    def test_null_end_date(self):
        e = CareerEntry(end_date=None)
        assert e.end_date is None

    def test_null_string_date(self):
        e = CareerEntry(start_date="null", end_date="")
        assert e.start_date is None
        assert e.end_date is None

    def test_malformed_date_becomes_none(self):
        e = CareerEntry(start_date="not-a-date")
        assert e.start_date is None

    def test_negative_duration_becomes_none(self):
        e = CareerEntry(duration_months=-3)
        assert e.duration_months is None

    def test_description_defaults_empty(self):
        e = CareerEntry()
        assert e.description == ""


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------


class TestSkill:
    def test_happy_path(self):
        s = Skill(name="Python", proficiency="advanced", endorsements=42)
        assert s.name == "Python"
        assert s.endorsements == 42

    def test_negative_endorsements_clamped(self):
        s = Skill(name="Go", endorsements=-5)
        assert s.endorsements == 0

    def test_none_duration_allowed(self):
        s = Skill(name="Rust", duration_months=None)
        assert s.duration_months is None

    def test_null_name_becomes_empty(self):
        s = Skill(name=None)
        assert s.name == ""


# ---------------------------------------------------------------------------
# Certification
# ---------------------------------------------------------------------------


class TestCertification:
    def test_happy_path(self):
        c = Certification(name="AWS Cloud Practitioner", issuer="AWS", year=2024)
        assert c.year == 2024

    def test_null_year(self):
        c = Certification(name="CKA", year=None)
        assert c.year is None

    def test_string_year_coerced(self):
        c = Certification(name="CKA", year="2023")
        assert c.year == 2023


# ---------------------------------------------------------------------------
# RedrobSignals
# ---------------------------------------------------------------------------


class TestRedrobSignals:
    def test_date_parsing(self):
        from datetime import date
        s = RedrobSignals(signup_date="2022-03-10", last_active_date="2024-01-01")
        assert s.signup_date == date(2022, 3, 10)

    def test_null_date(self):
        s = RedrobSignals(signup_date=None)
        assert s.signup_date is None

    def test_unknown_extra_fields_allowed(self):
        # extra="allow" in Config
        s = RedrobSignals(future_signal=True)
        assert s.future_signal is True


# ---------------------------------------------------------------------------
# Candidate (root model)
# ---------------------------------------------------------------------------


class TestCandidate:
    def _minimal(self) -> dict:
        return {"candidate_id": "CAND_0000001"}

    def test_minimal_valid(self):
        c = Candidate(**self._minimal())
        assert c.candidate_id == "CAND_0000001"
        assert c.skills == []
        assert c.certifications == []
        assert c.career_history == []

    def test_missing_candidate_id_raises(self):
        with pytest.raises(ValidationError):
            Candidate(candidate_id="")

    def test_empty_skills_filtered(self):
        data = {
            "candidate_id": "CAND_X",
            "skills": [
                {"name": "Python", "proficiency": "advanced"},
                {"name": "", "proficiency": "beginner"},   # empty name
                {"name": "   ", "proficiency": "beginner"}, # whitespace-only
            ],
        }
        c = Candidate(**data)
        assert len(c.skills) == 1
        assert c.skills[0].name == "Python"

    def test_null_profile_block_defaults(self):
        c = Candidate.model_validate({"candidate_id": "CAND_Y", "profile": None})
        assert c.profile.headline == ""

    def test_null_career_history_defaults(self):
        c = Candidate.model_validate({"candidate_id": "CAND_Z", "career_history": None})
        assert c.career_history == []

    def test_full_candidate_round_trip(self):
        import json, pathlib
        # Use first line of real data if available
        data_path = pathlib.Path(__file__).parents[1] / "data" / "candidates.jsonl"
        if not data_path.exists():
            pytest.skip("data/candidates.jsonl not found — skipping integration check")
        with open(data_path) as fh:
            raw = json.loads(fh.readline())
        c = Candidate.model_validate(raw)
        assert c.candidate_id.startswith("CAND_")
