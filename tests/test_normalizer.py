"""
tests/test_normalizer.py
========================
Unit tests for CandidateNormalizer.

Checks:
- normalized_text is non-empty for a full candidate
- Section headers appear correctly
- Empty sections are omitted
- Skill names joined correctly
- Cert formatting (name, issuer, year)
- JSON list fields are valid JSON
- Counts match input data
"""

from __future__ import annotations

import json

import pytest

from src.ingestion.models import (
    Candidate,
    CareerEntry,
    CandidateProfile,
    Certification,
    Language,
    RedrobSignals,
    Skill,
)
from src.ingestion.normalizer import CandidateNormalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(**overrides) -> Candidate:
    defaults = dict(
        candidate_id="CAND_TEST_001",
        profile=CandidateProfile(
            headline="ML Engineer",
            summary="10 years building ML systems",
            years_of_experience=10.0,
            current_title="Senior ML Engineer",
            current_company="Acme Corp",
        ),
        career_history=[
            CareerEntry(
                company="Acme Corp",
                title="Senior ML Engineer",
                description="Built recommendation systems using collaborative filtering.",
                is_current=True,
            ),
            CareerEntry(
                company="StartupXYZ",
                title="Data Scientist",
                description="Analysed petabytes of clickstream data.",
                is_current=False,
            ),
        ],
        skills=[
            Skill(name="Python", proficiency="advanced", endorsements=50),
            Skill(name="PyTorch", proficiency="advanced", endorsements=30),
        ],
        certifications=[
            Certification(name="AWS Cloud Practitioner", issuer="AWS", year=2023),
            Certification(name="TensorFlow Developer", issuer="Google"),
        ],
        languages=[Language(language="English", proficiency="native")],
        redrob_signals=RedrobSignals(
            notice_period_days=30,
            open_to_work_flag=True,
            github_activity_score=0.85,
        ),
    )
    defaults.update(overrides)
    return Candidate(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCandidateNormalizerText:
    def setup_method(self):
        self.normalizer = CandidateNormalizer()

    def test_headline_section_present(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert "[HEADLINE]" in nc.normalized_text
        assert "ML Engineer" in nc.normalized_text

    def test_summary_section_present(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert "[SUMMARY]" in nc.normalized_text
        assert "10 years building ML systems" in nc.normalized_text

    def test_experience_section_present(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert "[EXPERIENCE]" in nc.normalized_text
        assert "recommendation systems" in nc.normalized_text
        assert "clickstream" in nc.normalized_text

    def test_skills_section_present(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert "[SKILLS]" in nc.normalized_text
        assert "Python" in nc.normalized_text
        assert "PyTorch" in nc.normalized_text

    def test_certs_section_present(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert "[CERTS]" in nc.normalized_text
        assert "AWS Cloud Practitioner" in nc.normalized_text
        assert "AWS" in nc.normalized_text
        assert "2023" in nc.normalized_text

    def test_empty_headline_omits_section(self):
        c = _make_candidate()
        c.profile.headline = ""
        nc = self.normalizer.normalize(c)
        assert "[HEADLINE]" not in nc.normalized_text

    def test_no_skills_omits_section(self):
        c = _make_candidate(skills=[])
        nc = self.normalizer.normalize(c)
        assert "[SKILLS]" not in nc.normalized_text

    def test_no_certs_omits_section(self):
        c = _make_candidate(certifications=[])
        nc = self.normalizer.normalize(c)
        assert "[CERTS]" not in nc.normalized_text

    def test_cert_without_year(self):
        nc = self.normalizer.normalize(_make_candidate())
        # TensorFlow Developer has no year
        assert "TensorFlow Developer (Google)" in nc.normalized_text


class TestCandidateNormalizerStructuredFields:
    def setup_method(self):
        self.normalizer = CandidateNormalizer()

    def test_skill_names_json_valid(self):
        nc = self.normalizer.normalize(_make_candidate())
        names = json.loads(nc.skill_names_json)
        assert "Python" in names
        assert "PyTorch" in names

    def test_career_titles_json_valid(self):
        nc = self.normalizer.normalize(_make_candidate())
        titles = json.loads(nc.career_titles_json)
        assert "Senior ML Engineer" in titles

    def test_career_descriptions_json_valid(self):
        nc = self.normalizer.normalize(_make_candidate())
        descs = json.loads(nc.career_descriptions_json)
        assert any("recommendation" in d for d in descs)

    def test_counts_correct(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert nc.num_skills == 2
        assert nc.num_certifications == 2
        assert nc.num_career_entries == 2
        assert nc.num_languages == 1

    def test_signals_mapped(self):
        nc = self.normalizer.normalize(_make_candidate())
        assert nc.notice_period_days == 30
        assert nc.open_to_work_flag is True
        assert abs(nc.github_activity_score - 0.85) < 1e-6

    def test_empty_candidate_does_not_crash(self):
        c = Candidate(candidate_id="CAND_EMPTY")
        nc = self.normalizer.normalize(c)
        assert nc.normalized_text == ""
        assert nc.num_skills == 0
        assert json.loads(nc.skill_names_json) == []


class TestNormalizeBatch:
    def test_batch_length(self):
        normalizer = CandidateNormalizer()
        candidates = [_make_candidate(candidate_id=f"CAND_{i:04d}") for i in range(10)]
        results = normalizer.normalize_batch(candidates)
        assert len(results) == 10
        assert results[0].candidate_id == "CAND_0000"
