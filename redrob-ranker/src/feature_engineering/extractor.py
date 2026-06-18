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
        signals = candidate.redrob_signals.model_dump(exclude_none=True)
        for k, v in signals.items():
            if isinstance(v, (int, float, bool, str)): # Keep simple scalar features
                features[f"signal_{k}"] = v
                
        # 4. Location Features
        features["location"] = candidate.profile.location or ""
        features["country"] = candidate.profile.country or ""
        
        # 5. Notice Period
        features["notice_period_days"] = candidate.redrob_signals.notice_period_days or 0
        
        return features
