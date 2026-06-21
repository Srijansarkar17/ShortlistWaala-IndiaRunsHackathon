"""
CandidateNormalizer
===================
Converts a validated `Candidate` model into a flat, searchable text document
plus a structured record suitable for Parquet storage.

Normalized text document layout
--------------------------------
[HEADLINE]   <profile.headline>
[SUMMARY]    <profile.summary>
[EXPERIENCE] <career_history[0].description> ... <career_history[n].description>
[SKILLS]     <skill1>, <skill2>, ...
[CERTS]      <cert1 (issuer, year)>, ...

The flat record mirrors what gets written to Parquet (all scalar or JSON-encoded
list columns for downstream feature engineering).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .models import Candidate


_SECTION_SEP = "\n\n"


@dataclass
class NormalizedCandidate:
    """Flat output record ready for Parquet export."""

    # Identity
    candidate_id: str
    anonymized_name: str

    # Normalised free-text document (used for embedding / BM25)
    normalized_text: str

    # Structured scalars from profile
    headline: str
    summary: str
    location: Optional[str]
    country: Optional[str]
    years_of_experience: Optional[float]
    current_title: Optional[str]
    current_company: Optional[str]
    current_company_size: Optional[str]
    current_industry: Optional[str]

    # Aggregated / serialised list fields (JSON strings for easy Parquet storage)
    skill_names_json: str          # JSON array of skill name strings
    certification_names_json: str  # JSON array of cert name strings
    career_titles_json: str        # JSON array of job titles (ordered)
    career_companies_json: str     # JSON array of companies (ordered)
    career_descriptions_json: str  # JSON array of descriptions (ordered)

    # Counts
    num_skills: int
    num_certifications: int
    num_career_entries: int
    num_languages: int

    # Platform signals (key scalars surfaced for later feature engineering)
    profile_completeness_score: Optional[float]
    notice_period_days: Optional[int]
    open_to_work_flag: Optional[bool]
    github_activity_score: Optional[float]
    willing_to_relocate: Optional[bool]
    preferred_work_mode: Optional[str]
    verified_email: Optional[bool]
    verified_phone: Optional[bool]


class CandidateNormalizer:
    """
    Transforms a `Candidate` into a `NormalizedCandidate`.

    Usage::

        normalizer = CandidateNormalizer()
        record = normalizer.normalize(candidate)
    """

    def normalize(self, candidate: Candidate) -> NormalizedCandidate:
        """Return a flat `NormalizedCandidate` from a validated `Candidate`."""
        profile = candidate.profile
        history = candidate.career_history
        skills = candidate.skills
        certs = candidate.certifications
        signals = candidate.redrob_signals

        # ---- text sections -----------------------------------------------
        headline_text = self._clean(profile.headline)
        summary_text = self._clean(profile.summary)
        experience_text = self._build_experience(history)
        skills_text = self._build_skills(skills)
        certs_text = self._build_certs(certs)

        parts = [
            f"[HEADLINE] {headline_text}" if headline_text else "",
            f"[SUMMARY] {summary_text}" if summary_text else "",
            f"[EXPERIENCE] {experience_text}" if experience_text else "",
            f"[SKILLS] {skills_text}" if skills_text else "",
            f"[CERTS] {certs_text}" if certs_text else "",
        ]
        normalized_text = _SECTION_SEP.join(p for p in parts if p).strip()

        # ---- structured lists --------------------------------------------
        skill_names = [s.name for s in skills if s.name]
        cert_names = [c.name for c in certs if c.name]
        career_titles = [h.title for h in history]
        career_companies = [h.company for h in history]
        career_descriptions = [h.description for h in history]

        return NormalizedCandidate(
            candidate_id=candidate.candidate_id,
            anonymized_name=self._clean(profile.anonymized_name),
            normalized_text=normalized_text,
            # profile scalars
            headline=headline_text,
            summary=summary_text,
            location=profile.location,
            country=profile.country,
            years_of_experience=profile.years_of_experience,
            current_title=profile.current_title,
            current_company=profile.current_company,
            current_company_size=profile.current_company_size,
            current_industry=profile.current_industry,
            # serialised lists
            skill_names_json=json.dumps(skill_names, ensure_ascii=False),
            certification_names_json=json.dumps(cert_names, ensure_ascii=False),
            career_titles_json=json.dumps(career_titles, ensure_ascii=False),
            career_companies_json=json.dumps(career_companies, ensure_ascii=False),
            career_descriptions_json=json.dumps(career_descriptions, ensure_ascii=False),
            # counts
            num_skills=len(skills),
            num_certifications=len(certs),
            num_career_entries=len(history),
            num_languages=len(candidate.languages),
            # signals
            profile_completeness_score=signals.profile_completeness_score,
            notice_period_days=signals.notice_period_days,
            open_to_work_flag=signals.open_to_work_flag,
            github_activity_score=signals.github_activity_score,
            willing_to_relocate=signals.willing_to_relocate,
            preferred_work_mode=signals.preferred_work_mode,
            verified_email=signals.verified_email,
            verified_phone=signals.verified_phone,
        )

    def normalize_batch(self, candidates: list[Candidate]) -> list[NormalizedCandidate]:
        """Normalize a list of candidates."""
        return [self.normalize(c) for c in candidates]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Strip whitespace and collapse internal runs."""
        return " ".join(text.split()) if text else ""

    @staticmethod
    def _build_experience(history: list) -> str:
        """Join all career descriptions separated by a delimiter."""
        descriptions = [
            h.description.strip()
            for h in history
            if h.description and h.description.strip()
        ]
        return " | ".join(descriptions)

    @staticmethod
    def _build_skills(skills: list) -> str:
        """Comma-joined skill names."""
        return ", ".join(s.name for s in skills if s.name)

    @staticmethod
    def _build_certs(certs: list) -> str:
        """Readable cert list: 'Name (Issuer, Year)'."""
        parts = []
        for c in certs:
            if not c.name:
                continue
            meta_parts = []
            if c.issuer:
                meta_parts.append(c.issuer)
            if c.year:
                meta_parts.append(str(c.year))
            if meta_parts:
                parts.append(f"{c.name} ({', '.join(meta_parts)})")
            else:
                parts.append(c.name)
        return "; ".join(parts)
