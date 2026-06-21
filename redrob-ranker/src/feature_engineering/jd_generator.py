from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from loguru import logger
from src.jd_understanding.parser import JDProfile

class JDFeatureGenerator:
    """Generates features comparing candidates to a specific Job Description Profile."""

    def generate(
        self,
        jd_profile: JDProfile,
        candidates_parquet: str | Path,
        features_parquet: str | Path,
        honeypots_parquet: str | Path,
        retrieval_parquet: str | Path,
        filter_honeypots: bool = False
    ) -> pd.DataFrame:
        """
        Loads candidate data and generates match features relative to the JD profile.
        
        Parameters
        ----------
        jd_profile : JDProfile
            The parsed Job Description profile.
        candidates_parquet : str | Path
            Path to the normalized candidates database.
        features_parquet : str | Path
            Path to the candidate precomputed feature store.
        honeypots_parquet : str | Path
            Path to the honeypot detection results.
        retrieval_parquet : str | Path
            Path to the hybrid retrieval results.
        filter_honeypots : bool
            If True, candidates flagged as honeypots are omitted from output features.
        """
        logger.info("Loading candidates, precomputed features, honeypots, and retrieval results...")
        
        # Load all tables
        candidates_df = pd.read_parquet(candidates_parquet)
        features_df = pd.read_parquet(features_parquet)
        honeypots_df = pd.read_parquet(honeypots_parquet)
        retrieval_df = pd.read_parquet(retrieval_parquet)
        
        # Perform joins on candidate_id
        # We start with the full pool of candidates
        combined_df = candidates_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_f"))
        combined_df = combined_df.merge(honeypots_df, on="candidate_id", how="inner", suffixes=("", "_h"))
        
        # Join with retrieval results (using left join since retrieval might only contain top N)
        combined_df = combined_df.merge(retrieval_df, on="candidate_id", how="left", suffixes=("", "_r"))
        
        logger.info(f"Loaded and joined data for {len(combined_df):,} candidates.")
        
        # Filter out honeypots if requested
        if filter_honeypots:
            before_cnt = len(combined_df)
            combined_df = combined_df[combined_df["is_honeypot"] == False].reset_index(drop=True)
            removed = before_cnt - len(combined_df)
            logger.info(f"Filtered out {removed:,} honeypot candidates (remaining: {len(combined_df):,}).")
            
        # Initialize output feature records
        records = []
        
        # Lowercase search lists for fast comparisons
        req_skills_lower = [s.lower() for s in jd_profile.required_skills]
        pref_skills_lower = [s.lower() for s in jd_profile.preferred_skills]
        jd_locs_lower = [l.lower() for l in jd_profile.location_requirements]
        jd_industries_lower = [i.lower() for i in jd_profile.industry_requirements]
        all_terms_lower = [t.lower() for t in (jd_profile.required_skills + jd_profile.soft_skills)]
        
        # Experience boundaries
        exp_min = jd_profile.experience_range.min_years
        exp_max = jd_profile.experience_range.max_years
        
        # Loop through each row to construct relative features
        # Using itertuples is efficient in Pandas
        for row in combined_df.itertuples():
            candidate_id = row.candidate_id
            
            # --- 1. Skill Match ---
            skill_names = []
            try:
                if isinstance(row.skill_names_json, str):
                    skill_names = [s.lower() for s in json.loads(row.skill_names_json)]
            except Exception:
                pass
                
            skill_set = set(skill_names)
            
            req_matched = [s for s in req_skills_lower if s in skill_set]
            pref_matched = [s for s in pref_skills_lower if s in skill_set]
            
            req_ratio = len(req_matched) / len(req_skills_lower) if req_skills_lower else 1.0
            pref_ratio = len(pref_matched) / len(pref_skills_lower) if pref_skills_lower else 1.0
            
            # Weighted skill match score (70% required, 30% preferred)
            skill_match_score = (0.7 * req_ratio) + (0.3 * pref_ratio)
            
            # --- 2. Experience Match ---
            yoe = getattr(row, "years_of_experience", 0.0)
            if pd.isna(yoe) or yoe is None:
                yoe = 0.0
                
            min_satisfied = 1.0
            max_satisfied = 1.0
            
            if exp_min is not None and exp_min > 0:
                if yoe >= exp_min:
                    min_satisfied = 1.0
                else:
                    min_satisfied = max(0.0, 1.0 - (exp_min - yoe) * 0.5)
                    
            if exp_max is not None and exp_max > 0:
                if yoe <= exp_max:
                    max_satisfied = 1.0
                else:
                    # Steeper penalty for overqualification
                    max_satisfied = max(0.0, 1.0 - (yoe - exp_max) * 0.5)
                    
            exp_match_score = (min_satisfied + max_satisfied) / 2.0

            # --- 2.5 Role Relevance Check ---
            cand_headline = (getattr(row, "headline", "") or "").lower()
            cand_curr_title = (getattr(row, "current_title", "") or "").lower()
            
            disqualified_kws = [
                "civil", "hr", "human resources", "mechanical", "graphic designer", 
                "accountant", "accounting", "operations", "customer support", 
                "content writer", "marketing", "sales", "brand manager", "seo", 
                "copywriter", "social media", "public relations", "ad operations",
                "finance", "legal", "teacher", "professor", "doctor", "nurse",
                "recruiter", "talent acquisition", "project manager", "devops", 
                "qa", "quality assurance", "testing", "test engineer", "frontend", 
                "front-end", "front end", "ui/ux", "mobile developer", "android", 
                "ios", "scrum master", "analyst"
            ]
            
            tech_kws = [
                "software", "machine learning", "ai", "ml", "nlp", "deep learning", 
                "computer vision", "data scientist", "data science", "data engineer", 
                "developer", "programmer", "tech lead", "technical lead", "architect", 
                "systems engineer", "engineering", "coder", "backend", "front-end", 
                "frontend", "fullstack", "full-stack", "back-end"
            ]
            
            is_disqualified = any(kw in cand_headline for kw in disqualified_kws) or any(kw in cand_curr_title for kw in disqualified_kws)
            has_tech_kw = any(kw in cand_headline for kw in tech_kws) or any(kw in cand_curr_title for kw in tech_kws)
            feat_is_technical_role = 1.0 if (has_tech_kw and not is_disqualified) else 0.0

            # --- AI/ML/Data Science Specific Check ---
            import re
            def check_ai_ml(text: str) -> bool:
                text_l = text.lower()
                words = set(re.split(r'\W+', text_l))
                # Multi-word phrases
                phrases = ["machine learning", "deep learning", "natural language", "data scientist", "data science", "recommendation", "retrieval", "vector search"]
                if any(phrase in text_l for phrase in phrases):
                    return True
                # Words
                target_words = {"ai", "ml", "nlp", "ranking", "search"}
                if words.intersection(target_words):
                    return True
                return False

            is_ai_ml = check_ai_ml(cand_headline) or check_ai_ml(cand_curr_title)
            feat_is_ai_ml_role = 1.0 if is_ai_ml else 0.0

            
            # --- 3. Location Match ---
            cand_loc = (getattr(row, "location", "") or "").lower()
            cand_country = (getattr(row, "country", "") or "").lower()
            cand_mode = (getattr(row, "preferred_work_mode", "") or "").lower()
            cand_reloc = bool(getattr(row, "willing_to_relocate", False))
            
            loc_match_score = 0.0
            if not jd_locs_lower:
                loc_match_score = 1.0
            else:
                # Check direct word overlap with candidate location
                direct_match = any(loc in cand_loc for loc in jd_locs_lower)
                in_india = ("india" in cand_country) or ("india" in cand_loc)
                jd_has_india = any("india" in loc for loc in jd_locs_lower)
                
                if direct_match:
                    loc_match_score = 1.0
                elif cand_reloc and (in_india or not jd_has_india):
                    loc_match_score = 0.8
                elif "remote" in cand_mode and any("remote" in loc for loc in jd_locs_lower):
                    loc_match_score = 0.8
                elif "hybrid" in cand_mode and any("hybrid" in loc for loc in jd_locs_lower):
                    loc_match_score = 0.6
                elif cand_reloc:
                    loc_match_score = 0.5
                    
            # --- 4. Domain Match ---
            cand_ind = (getattr(row, "current_industry", "") or "").lower()
            
            career_industries = []
            try:
                # Fallback check for any listed industry terms in career descriptions or titles
                desc_json = getattr(row, "career_descriptions_json", "[]")
                if isinstance(desc_json, str):
                    career_industries = json.loads(desc_json)
            except Exception:
                pass
            
            desc_text = " ".join(career_industries).lower()
            
            domain_match_score = 0.0
            if not jd_industries_lower:
                domain_match_score = 1.0
            else:
                matches_current = any(ind in cand_ind for ind in jd_industries_lower)
                matches_history = any(ind in desc_text for ind in jd_industries_lower)
                if matches_current:
                    domain_match_score = 1.0
                elif matches_history:
                    domain_match_score = 0.7
                    
            # --- 5. Responsibility Match ---
            titles = []
            try:
                titles_json = getattr(row, "career_titles_json", "[]")
                if isinstance(titles_json, str):
                    titles = json.loads(titles_json)
            except Exception:
                pass
                
            title_text = " ".join(titles).lower()
            
            matched_terms = 0
            for term in all_terms_lower:
                if term in desc_text or term in title_text:
                    matched_terms += 1
            
            responsibility_match_score = (
                matched_terms / len(all_terms_lower) if all_terms_lower else 1.0
            )
            
            # --- 6. Certification Match ---
            certs = []
            try:
                certs_json = getattr(row, "certification_names_json", "[]")
                if isinstance(certs_json, str):
                    certs = json.loads(certs_json)
            except Exception:
                pass
                
            cert_text = " ".join(certs).lower()
            
            cert_match_score = 0.0
            if certs:
                # Match certifications against required skills
                cert_matches = [s for s in req_skills_lower if s in cert_text]
                if cert_matches:
                    cert_match_score = 1.0
                else:
                    cert_match_score = 0.2  # baseline for having certifications
                    
            # --- 7 & 8. Semantic Match & Retrieval Match ---
            # Retrieval results are from Phase 5 output
            dense_score = getattr(row, "dense_score", None)
            bm25_score = getattr(row, "bm25_score", None)
            rrf_score = getattr(row, "rrf_score", None)
            combined_rank = getattr(row, "combined_rank", None)
            
            retrieved = 1.0 if not pd.isna(rrf_score) else 0.0
            
            # Fill default values if not retrieved
            dense_score_val = float(dense_score) if retrieved else 0.0
            bm25_score_val = float(bm25_score) if retrieved else 0.0
            rrf_score_val = float(rrf_score) if retrieved else 0.0
            combined_rank_val = int(combined_rank) if retrieved else 999999
            
            # We map semantic match directly to dense cosine score
            semantic_match_score = dense_score_val
            
            # Assemble feature record
            records.append({
                "candidate_id": candidate_id,
                # Output match features
                "feat_required_skills_match_ratio": float(req_ratio),
                "feat_preferred_skills_match_ratio": float(pref_ratio),
                "feat_skills_match_score": float(skill_match_score),
                "feat_experience_match_score": float(exp_match_score),
                "feat_is_technical_role": float(feat_is_technical_role),
                "feat_is_ai_ml_role": float(feat_is_ai_ml_role),
                "feat_location_match_score": float(loc_match_score),
                "feat_domain_match_score": float(domain_match_score),
                "feat_responsibility_match_score": float(responsibility_match_score),
                "feat_certification_match_score": float(cert_match_score),
                "feat_semantic_match": float(semantic_match_score),
                # Retrieval features
                "feat_retrieved": float(retrieved),
                "feat_bm25_score": float(bm25_score_val),
                "feat_dense_score": float(dense_score_val),
                "feat_rrf_score": float(rrf_score_val),
                "feat_combined_rank": int(combined_rank_val),
                # Honeypot scores (useful for downstream sorting and penalty application)
                "honeypot_score": float(getattr(row, "honeypot_score", 0.0)),
                "trust_score": float(getattr(row, "trust_score", 1.0)),
                "is_honeypot": bool(getattr(row, "is_honeypot", False))
            })
            
        features_df = pd.DataFrame(records)
        return features_df
