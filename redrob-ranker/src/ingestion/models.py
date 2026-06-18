"""
Pydantic v2 dataclass models for candidate profiles.

All fields are typed defensively:
 - Optional fields default to None / empty collections
 - Validators clean common malformed values before validation fails
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CandidateProfile(BaseModel):
    """Top-level profile block."""

    anonymized_name: str = Field(default="")
    headline: str = Field(default="")
    summary: str = Field(default="")
    location: Optional[str] = None
    country: Optional[str] = None
    years_of_experience: Optional[float] = None
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    current_company_size: Optional[str] = None
    current_industry: Optional[str] = None

    @field_validator("years_of_experience", mode="before")
    @classmethod
    def coerce_yoe(cls, v: Any) -> Optional[float]:
        """Accept int / str representations, map negatives to None."""
        if v is None:
            return None
        try:
            val = float(v)
            return val if val >= 0 else None
        except (TypeError, ValueError):
            return None

    @field_validator("anonymized_name", "headline", "summary", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


class CareerEntry(BaseModel):
    """Single job / role in career history."""

    company: str = Field(default="")
    title: str = Field(default="")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    industry: Optional[str] = None
    company_size: Optional[str] = None
    description: str = Field(default="")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_nullable_date(cls, v: Any) -> Optional[date]:
        """Accept ISO string, date object, or None."""
        if v is None or v == "" or v == "null":
            return None
        if isinstance(v, date):
            return v
        try:
            return date.fromisoformat(str(v)[:10])
        except (ValueError, TypeError):
            return None

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            val = int(float(v))
            return val if val >= 0 else None
        except (TypeError, ValueError):
            return None

    @field_validator("company", "title", "description", mode="before")
    @classmethod
    def strip_str(cls, v: Any) -> str:
        return str(v).strip() if v else ""


class Education(BaseModel):
    """Education entry."""

    institution: str = Field(default="")
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    grade: Optional[str] = None
    tier: Optional[str] = None

    @field_validator("start_year", "end_year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None


class Skill(BaseModel):
    """Individual skill with proficiency metadata."""

    name: str = Field(default="")
    proficiency: Optional[str] = None
    endorsements: int = Field(default=0)
    duration_months: Optional[int] = None

    @field_validator("name", mode="before")
    @classmethod
    def require_name(cls, v: Any) -> str:
        return str(v).strip() if v else ""

    @field_validator("endorsements", mode="before")
    @classmethod
    def coerce_endorsements(cls, v: Any) -> int:
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return 0

    @field_validator("duration_months", mode="before")
    @classmethod
    def coerce_duration(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return max(0, int(v))
        except (TypeError, ValueError):
            return None


class Certification(BaseModel):
    """Professional certification."""

    name: str = Field(default="")
    issuer: Optional[str] = None
    year: Optional[int] = None

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None


class Language(BaseModel):
    """Language proficiency entry."""

    language: str = Field(default="")
    proficiency: Optional[str] = None


class RedrobSignals(BaseModel):
    """Platform behavioral signals from RedRob."""

    profile_completeness_score: Optional[float] = None
    signup_date: Optional[date] = None
    last_active_date: Optional[date] = None
    open_to_work_flag: Optional[bool] = None
    profile_views_received_30d: Optional[int] = None
    applications_submitted_30d: Optional[int] = None
    recruiter_response_rate: Optional[float] = None
    avg_response_time_hours: Optional[float] = None
    skill_assessment_scores: Optional[dict[str, Any]] = None
    connection_count: Optional[int] = None
    endorsements_received: Optional[int] = None
    notice_period_days: Optional[int] = None
    expected_salary_range_inr_lpa: Optional[dict[str, Any]] = None
    preferred_work_mode: Optional[str] = None
    willing_to_relocate: Optional[bool] = None
    github_activity_score: Optional[float] = None
    search_appearance_30d: Optional[int] = None
    saved_by_recruiters_30d: Optional[int] = None
    interview_completion_rate: Optional[float] = None
    offer_acceptance_rate: Optional[float] = None
    verified_email: Optional[bool] = None
    verified_phone: Optional[bool] = None
    linkedin_connected: Optional[bool] = None

    @field_validator("signup_date", "last_active_date", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> Optional[date]:
        if v is None or v == "":
            return None
        if isinstance(v, date):
            return v
        try:
            return date.fromisoformat(str(v)[:10])
        except (ValueError, TypeError):
            return None

    model_config = ConfigDict(extra="allow")  # tolerate any future signal fields


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    """Root candidate model.  All nested lists default to empty."""

    candidate_id: str
    profile: CandidateProfile = Field(default_factory=CandidateProfile)
    career_history: list[CareerEntry] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)

    @field_validator("candidate_id", mode="before")
    @classmethod
    def require_id(cls, v: Any) -> str:
        if not v:
            raise ValueError("candidate_id must not be empty")
        return str(v).strip()

    @field_validator("skills", mode="before")
    @classmethod
    def drop_empty_skills(cls, v: Any) -> list:
        """Remove skill entries where name is explicitly blank/null."""
        if not isinstance(v, list):
            return []
        cleaned = []
        for s in v:
            if isinstance(s, dict):
                name = s.get("name")
                # Only drop if name is None, empty string, or pure whitespace
                if name is None or (isinstance(name, str) and not name.strip()):
                    continue
                cleaned.append(s)
            elif isinstance(s, Skill):
                if s.name and s.name.strip():
                    cleaned.append(s)
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def handle_missing_top_level(cls, values: Any) -> Any:
        """Guarantee all top-level keys exist even if absent from JSON."""
        if not isinstance(values, dict):
            return values
        defaults: dict[str, Any] = {
            "profile": {},
            "career_history": [],
            "education": [],
            "skills": [],
            "certifications": [],
            "languages": [],
            "redrob_signals": {},
        }
        for key, default in defaults.items():
            if values.get(key) is None:
                values[key] = default
        return values
