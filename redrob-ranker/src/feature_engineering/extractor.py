from datetime import date
import json
from typing import Any

from src.ingestion.models import Candidate

class FeatureExtractor:
    """Extracts features for Phase 3."""
    
    def extract(self, candidate: Candidate) -> dict[str, Any]:
        features: dict[str, Any] = {"candidate_id": candidate.candidate_id}
        
        # 1. Experience Features
        total_exp = candidate.profile.years_of_experience or 0.0
        features["total_experience"] = total_exp
        
        # Calculate relevant experience (simplified heuristic: same title keyword)
        relevant_months = 0
        current_title = candidate.profile.current_title.lower() if candidate.profile.current_title else ""
        
        total_tenure_months = 0
        promotion_count = 0
        company_roles = {}
        
        product_jobs = 0
        consulting_jobs = 0
        startup_jobs = 0
        
        num_jobs = len(candidate.career_history)
        
        for entry in candidate.career_history:
            dur = entry.duration_months or 0
            total_tenure_months += dur
            
            # Relevant experience heuristic
            if current_title and current_title in entry.title.lower():
                relevant_months += dur
                
            # Track promotions (multiple roles in same company)
            if entry.company:
                company_roles[entry.company] = company_roles.get(entry.company, 0) + 1
                
            # Company type ratios heuristic
            desc = (entry.description + " " + entry.company + " " + (entry.industry or "")).lower()
            if "product" in desc or "saas" in desc:
                product_jobs += 1
            if "consult" in desc or "services" in desc or "agency" in desc:
                consulting_jobs += 1
            if "startup" in desc or "founder" in desc or (entry.company_size and "1-10" in entry.company_size):
                startup_jobs += 1

        features["relevant_experience"] = relevant_months / 12.0
        features["avg_tenure"] = (total_tenure_months / num_jobs / 12.0) if num_jobs > 0 else 0.0
        
        for roles in company_roles.values():
            if roles > 1:
                promotion_count += (roles - 1)
        features["promotion_count"] = promotion_count
        
        # 2. Company Features
        features["product_company_ratio"] = product_jobs / num_jobs if num_jobs > 0 else 0.0
        features["consulting_ratio"] = consulting_jobs / num_jobs if num_jobs > 0 else 0.0
        features["startup_ratio"] = startup_jobs / num_jobs if num_jobs > 0 else 0.0
        
        # 3. Behavior Features (Redrob signals)
        signals = candidate.redrob_signals
        
        features["signal_profile_completeness_score"] = signals.profile_completeness_score
        features["signal_signup_date"] = str(signals.signup_date) if signals.signup_date else None
        features["signal_last_active_date"] = str(signals.last_active_date) if signals.last_active_date else None
        features["signal_open_to_work_flag"] = signals.open_to_work_flag
        features["signal_profile_views_received_30d"] = signals.profile_views_received_30d
        features["signal_applications_submitted_30d"] = signals.applications_submitted_30d
        features["signal_recruiter_response_rate"] = signals.recruiter_response_rate
        features["signal_avg_response_time_hours"] = signals.avg_response_time_hours
        
        # skill_assessment_scores (JSON string for nested compatibility, plus aggregates)
        features["signal_skill_assessment_scores_json"] = json.dumps(signals.skill_assessment_scores or {})
        
        # Aggregates for skill assessments
        features["signal_num_skill_assessments"] = len(signals.skill_assessment_scores) if signals.skill_assessment_scores else 0
        if signals.skill_assessment_scores:
            scores = [float(s) for s in signals.skill_assessment_scores.values() if s is not None]
            if scores:
                features["signal_avg_skill_assessment_score"] = sum(scores) / len(scores)
                features["signal_max_skill_assessment_score"] = max(scores)
                features["signal_min_skill_assessment_score"] = min(scores)
            else:
                features["signal_avg_skill_assessment_score"] = None
                features["signal_max_skill_assessment_score"] = None
                features["signal_min_skill_assessment_score"] = None
        else:
            features["signal_avg_skill_assessment_score"] = None
            features["signal_max_skill_assessment_score"] = None
            features["signal_min_skill_assessment_score"] = None

        features["signal_connection_count"] = signals.connection_count
        features["signal_endorsements_received"] = signals.endorsements_received
        features["signal_notice_period_days"] = signals.notice_period_days
        
        # expected_salary_range_inr_lpa (flatten min and max)
        sal = signals.expected_salary_range_inr_lpa
        features["signal_expected_salary_min"] = sal.get("min") if sal else None
        features["signal_expected_salary_max"] = sal.get("max") if sal else None
        
        features["signal_preferred_work_mode"] = signals.preferred_work_mode
        features["signal_willing_to_relocate"] = signals.willing_to_relocate
        features["signal_github_activity_score"] = signals.github_activity_score
        features["signal_search_appearance_30d"] = signals.search_appearance_30d
        features["signal_saved_by_recruiters_30d"] = signals.saved_by_recruiters_30d
        features["signal_interview_completion_rate"] = signals.interview_completion_rate
        features["signal_offer_acceptance_rate"] = signals.offer_acceptance_rate
        features["signal_verified_email"] = signals.verified_email
        features["signal_verified_phone"] = signals.verified_phone
        features["signal_linkedin_connected"] = signals.linkedin_connected
                
        # 4. Location Features
        features["location"] = candidate.profile.location or ""
        features["country"] = candidate.profile.country or ""
        
        # 5. Notice Period
        features["notice_period_days"] = candidate.redrob_signals.notice_period_days or 0
        
        return features


