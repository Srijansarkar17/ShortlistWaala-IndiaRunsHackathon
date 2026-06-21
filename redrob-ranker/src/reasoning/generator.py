from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from loguru import logger

class ReasoningGenerator:
    """Generates 1-2 sentence fact-grounded justifications for top candidate recommendations."""

    def generate_reasoning(self, candidate_row) -> str:
        """
        Creates a custom, grounded reasoning string for a candidate.
        """
        # 1. Experience & Title
        title = getattr(candidate_row, "headline", None) or getattr(candidate_row, "current_title", None) or "Software Engineer"
        title = title.strip()
        
        yoe = getattr(candidate_row, "years_of_experience", None)
        if yoe is None or pd.isna(yoe):
            yoe = getattr(candidate_row, "total_experience", 0.0) or 0.0
        
        yoe_str = f"{float(yoe):.1f}"

        # 2. Matched Skills
        skills_json = getattr(candidate_row, "skill_names_json", None)
        skills = []
        if skills_json:
            try:
                skills = json.loads(skills_json)
            except Exception:
                pass
        
        # Format a subset of skills (up to 3) for the text
        skills_to_show = [s for s in skills if s][:3]
        if skills_to_show:
            skills_phrase = ", ".join(skills_to_show)
        else:
            skills_phrase = "required technical skills"

        # Sentence 1: Profile suitability & experience fit
        prefix = "Senior " if float(yoe) >= 5.0 else ""
        s1 = f"{prefix}{title} with {yoe_str} years of experience, showing matching expertise in {skills_phrase}."

        # 3. Behavioral Signals & Gaps (Traps)
        response_rate = getattr(candidate_row, "signal_recruiter_response_rate", None)
        if response_rate is not None and not pd.isna(response_rate):
            rr_val = int(float(response_rate) * 100)
        else:
            rr_val = 0

        open_to_work = bool(getattr(candidate_row, "open_to_work_flag", False))
        notice_days = getattr(candidate_row, "notice_period_days", 0) or 0

        # Trap signals (gaps)
        job_hopper = float(getattr(candidate_row, "trap_job_hopper", 0.0) or 0.0)
        consulting = float(getattr(candidate_row, "trap_consulting_only", 0.0) or 0.0)
        inactive = float(getattr(candidate_row, "trap_inactive_candidate", 0.0) or 0.0)
        stuffer = float(getattr(candidate_row, "trap_keyword_stuffer", 0.0) or 0.0)

        # Build sentence 2
        s2_parts = []
        if open_to_work:
            s2_parts.append("Candidate is actively open to new work")
        else:
            s2_parts.append("Profile shows stable engagement")

        if rr_val > 50:
            s2_parts.append(f"a high recruiter response rate of {rr_val}%")
        elif rr_val > 0:
            s2_parts.append(f"a response rate of {rr_val}%")

        s2_base = " and ".join([p for p in s2_parts if p])
        if s2_base:
            s2 = f"{s2_base}."
        else:
            s2 = "Profile represents a solid fit with active platform presence."

        # Add gap/trap disclosures if applicable
        gaps = []
        if int(notice_days) > 60:
            gaps.append(f"a long notice period of {int(notice_days)} days")
        if job_hopper > 0.6:
            gaps.append("potential job-hopping history")
        if consulting > 0.7:
            gaps.append("exclusively service-firm consulting experience")
        if inactive > 0.6:
            gaps.append("low platform login recency")
        if stuffer > 0.6:
            gaps.append("potential keyword stuffing in profile skills")

        if gaps:
            gap_phrase = ", with ".join(gaps)
            # Combine sentences
            s2 = s2.rstrip(".") + f", though notes show {gap_phrase}."

        return f"{s1} {s2}"
