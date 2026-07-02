"""
src/validation/checks.py
========================
Deterministic honeypot checks — pure functions operating on a validated
`Candidate` object. Each check returns a `CheckResult` with:

  - triggered  : bool   — whether the anomaly was found
  - penalty    : float  — how much this deducts from trust_score (0–1 scale)
  - reason     : str    — human-readable explanation for logging / ranking

Check catalogue
---------------
  C1  salary_min > salary_max
  C2  job start_date > end_date (for any non-current role)
  C3  education start_year > end_year
  C4  declared duration_months vs date-computed duration mismatch > 3 months
  C5  "expert" / "advanced" skill with zero (0) duration_months, ≥3 such skills
  C6  any single skill duration > total_experience_months + 12
  C7  startup founding-year violation (worked at startup before it existed)

Penalty weights are additive; final score is clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from src.ingestion.models import Candidate


# ---------------------------------------------------------------------------
# Startup founding years
# ---------------------------------------------------------------------------

STARTUP_FOUNDING_YEARS: dict[str, int] = {
    "openai": 2015,
    "huggingface": 2016,
    "pinecone": 2019,
    "cohere": 2019,
    "anthropic": 2021,
    "qdrant": 2021,
    "perplexity": 2022,
    "langchain": 2022,
    "llamaindex": 2022,
    "mistral": 2023,
}

# ---------------------------------------------------------------------------
# Proficiency levels considered "expert" for check C5
# ---------------------------------------------------------------------------

EXPERT_PROFICIENCIES = {"expert", "advanced"}

# ---------------------------------------------------------------------------
# Penalty weights — tune here without touching logic
# ---------------------------------------------------------------------------

PENALTY_SALARY_INVERSION    = 0.30   # C1 — hard financial lie
PENALTY_JOB_DATE_INVERSION  = 0.25   # C2 — per inverted role, capped at 0.50
PENALTY_EDU_DATE_INVERSION  = 0.15   # C3 — per inverted entry, capped at 0.30
PENALTY_DURATION_MISMATCH   = 0.10   # C4 — per mismatch, capped at 0.40
PENALTY_EXPERT_ZERO_DUR     = 0.20   # C5 — batch of ≥3 zero-duration experts
PENALTY_SKILL_EXCEEDS_EXP   = 0.20   # C6 — per offending skill, capped at 0.40
PENALTY_STARTUP_VIOLATION   = 0.35   # C7 — per violated startup claim

DURATION_MISMATCH_TOLERANCE = 3      # months
EXPERT_ZERO_DUR_THRESHOLD   = 3      # minimum count to trigger C5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class CheckName(str, Enum):
    SALARY_INVERSION    = "C1_salary_inversion"
    JOB_DATE_INVERSION  = "C2_job_date_inversion"
    EDU_DATE_INVERSION  = "C3_edu_date_inversion"
    DURATION_MISMATCH   = "C4_duration_mismatch"
    EXPERT_ZERO_DUR     = "C5_expert_zero_duration"
    SKILL_EXCEEDS_EXP   = "C6_skill_exceeds_experience"
    STARTUP_VIOLATION   = "C7_startup_founding_violation"


@dataclass
class CheckResult:
    """Result for a single deterministic check."""

    check: CheckName
    triggered: bool
    penalty: float
    reason: str
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _months_between(start: date, end: date) -> int:
    """Approximate month difference (rounds down)."""
    return (end.year - start.year) * 12 + (end.month - start.month)


def check_salary_inversion(candidate: Candidate) -> CheckResult:
    """C1: salary_min > salary_max."""
    signals = candidate.redrob_signals
    salary = signals.expected_salary_range_inr_lpa

    if salary is None:
        return CheckResult(
            check=CheckName.SALARY_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="No salary data",
        )

    # salary may be a dict with 'min' and 'max' keys
    if isinstance(salary, dict):
        sal_min = salary.get("min")
        sal_max = salary.get("max")
    else:
        return CheckResult(
            check=CheckName.SALARY_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="Salary field not a dict",
        )

    if sal_min is None or sal_max is None:
        return CheckResult(
            check=CheckName.SALARY_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="Incomplete salary data",
        )

    try:
        sal_min = float(sal_min)
        sal_max = float(sal_max)
    except (TypeError, ValueError):
        return CheckResult(
            check=CheckName.SALARY_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="Non-numeric salary",
        )

    if sal_min > sal_max:
        return CheckResult(
            check=CheckName.SALARY_INVERSION,
            triggered=True,
            penalty=PENALTY_SALARY_INVERSION,
            reason=f"salary_min ({sal_min}) > salary_max ({sal_max})",
            details=[f"min={sal_min} LPA, max={sal_max} LPA"],
        )

    return CheckResult(
        check=CheckName.SALARY_INVERSION,
        triggered=False,
        penalty=0.0,
        reason="Salary range valid",
    )


def check_job_date_inversion(candidate: Candidate) -> CheckResult:
    """C2: any job where start_date > end_date (only non-current roles)."""
    violations = []
    for job in candidate.career_history:
        if job.is_current:
            continue
        if job.start_date and job.end_date and job.start_date > job.end_date:
            violations.append(
                f"{job.company} / {job.title}: {job.start_date} > {job.end_date}"
            )

    if not violations:
        return CheckResult(
            check=CheckName.JOB_DATE_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="All job date ranges valid",
        )

    penalty = min(len(violations) * PENALTY_JOB_DATE_INVERSION, 0.50)
    return CheckResult(
        check=CheckName.JOB_DATE_INVERSION,
        triggered=True,
        penalty=penalty,
        reason=f"{len(violations)} job(s) with start > end date",
        details=violations,
    )


def check_education_date_inversion(candidate: Candidate) -> CheckResult:
    """C3: education entry where start_year > end_year."""
    violations = []
    for edu in candidate.education:
        if edu.start_year and edu.end_year and edu.start_year > edu.end_year:
            violations.append(
                f"{edu.institution} ({edu.degree}): {edu.start_year} > {edu.end_year}"
            )

    if not violations:
        return CheckResult(
            check=CheckName.EDU_DATE_INVERSION,
            triggered=False,
            penalty=0.0,
            reason="All education date ranges valid",
        )

    penalty = min(len(violations) * PENALTY_EDU_DATE_INVERSION, 0.30)
    return CheckResult(
        check=CheckName.EDU_DATE_INVERSION,
        triggered=True,
        penalty=penalty,
        reason=f"{len(violations)} education entry/entries with start > end year",
        details=violations,
    )


def check_duration_mismatch(candidate: Candidate) -> CheckResult:
    """C4: declared duration_months differs from date-computed duration by > 3 months."""
    violations = []
    for job in candidate.career_history:
        if job.duration_months is None:
            continue
        if job.start_date is None or (job.end_date is None and not job.is_current):
            continue

        end = job.end_date if not job.is_current else date.today()
        if job.start_date > end:
            continue  # date inversion already caught by C2

        computed = _months_between(job.start_date, end)
        diff = abs(computed - job.duration_months)
        if diff > DURATION_MISMATCH_TOLERANCE:
            violations.append(
                f"{job.company} / {job.title}: declared={job.duration_months}m "
                f"computed={computed}m diff={diff}m"
            )

    if not violations:
        return CheckResult(
            check=CheckName.DURATION_MISMATCH,
            triggered=False,
            penalty=0.0,
            reason="All declared durations consistent with dates",
        )

    penalty = min(len(violations) * PENALTY_DURATION_MISMATCH, 0.40)
    return CheckResult(
        check=CheckName.DURATION_MISMATCH,
        triggered=True,
        penalty=penalty,
        reason=f"{len(violations)} job(s) with duration mismatch > {DURATION_MISMATCH_TOLERANCE} months",
        details=violations,
    )


def check_expert_zero_duration(candidate: Candidate) -> CheckResult:
    """C5: ≥3 'expert'/'advanced' skills with duration_months == 0."""
    zero_dur_experts = [
        s for s in candidate.skills
        if (s.proficiency or "").lower() in EXPERT_PROFICIENCIES
        and s.duration_months == 0
    ]

    if len(zero_dur_experts) < EXPERT_ZERO_DUR_THRESHOLD:
        return CheckResult(
            check=CheckName.EXPERT_ZERO_DUR,
            triggered=False,
            penalty=0.0,
            reason=f"Only {len(zero_dur_experts)} expert skills with zero duration (threshold={EXPERT_ZERO_DUR_THRESHOLD})",
        )

    return CheckResult(
        check=CheckName.EXPERT_ZERO_DUR,
        triggered=True,
        penalty=PENALTY_EXPERT_ZERO_DUR,
        reason=f"{len(zero_dur_experts)} expert/advanced skills claim zero duration",
        details=[f"{s.name} ({s.proficiency}, {s.duration_months}m)" for s in zero_dur_experts],
    )


def check_skill_exceeds_experience(candidate: Candidate) -> CheckResult:
    """C6: any skill duration > total_experience_months + 12."""
    yoe = candidate.profile.years_of_experience
    if yoe is None:
        return CheckResult(
            check=CheckName.SKILL_EXCEEDS_EXP,
            triggered=False,
            penalty=0.0,
            reason="No years_of_experience — check skipped",
        )

    total_exp_months = int(yoe * 12)
    ceiling = total_exp_months + 12

    violations = [
        s for s in candidate.skills
        if s.duration_months is not None and s.duration_months > ceiling
    ]

    if not violations:
        return CheckResult(
            check=CheckName.SKILL_EXCEEDS_EXP,
            triggered=False,
            penalty=0.0,
            reason=f"All skill durations ≤ experience ceiling ({ceiling}m)",
        )

    penalty = min(len(violations) * PENALTY_SKILL_EXCEEDS_EXP, 0.40)
    return CheckResult(
        check=CheckName.SKILL_EXCEEDS_EXP,
        triggered=True,
        penalty=penalty,
        reason=f"{len(violations)} skill(s) with duration > experience+12m (ceiling={ceiling}m)",
        details=[
            f"{s.name}: {s.duration_months}m vs ceiling {ceiling}m"
            for s in violations
        ],
    )


def check_startup_founding_year(candidate: Candidate) -> CheckResult:
    """C7: candidate's job start_date predates the startup's founding year."""
    violations = []
    for job in candidate.career_history:
        company_lower = job.company.lower().strip()
        for startup_name, founding_year in STARTUP_FOUNDING_YEARS.items():
            if startup_name in company_lower:
                if job.start_date and job.start_date.year < founding_year:
                    violations.append(
                        f"Claimed to work at {job.company} from {job.start_date} "
                        f"but {startup_name.title()} was founded in {founding_year}"
                    )
                break  # only match the first startup per job entry

    if not violations:
        return CheckResult(
            check=CheckName.STARTUP_VIOLATION,
            triggered=False,
            penalty=0.0,
            reason="No startup founding-year violations",
        )

    penalty = min(len(violations) * PENALTY_STARTUP_VIOLATION, 0.70)
    return CheckResult(
        check=CheckName.STARTUP_VIOLATION,
        triggered=True,
        penalty=penalty,
        reason=f"{len(violations)} startup(s) claim pre-dates founding year",
        details=violations,
    )


# ---------------------------------------------------------------------------
# Registry — ordered list of all checks
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_salary_inversion,
    check_job_date_inversion,
    check_education_date_inversion,
    check_duration_mismatch,
    check_expert_zero_duration,
    check_skill_exceeds_experience,
    check_startup_founding_year,
]
