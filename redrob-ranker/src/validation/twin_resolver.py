from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

class TwinResolver:
    """Identifies and resolves duplicate candidate profiles (twins) using behavioral strength."""

    def resolve(
        self,
        candidates_parquet: str | Path,
        features_parquet: str | Path,
        reference_date: datetime = datetime(2026, 6, 21)
    ) -> pd.DataFrame:
        """
        Loads candidate data, identifies twins, and generates a twin resolution score.
        
        Parameters
        ----------
        candidates_parquet : str | Path
            Path to the normalized candidates database.
        features_parquet : str | Path
            Path to the candidate precomputed feature store.
        reference_date : datetime
            The evaluation reference date for calculating inactivity.
        """
        logger.info("Loading candidates and features data for twin resolution...")
        
        candidates_df = pd.read_parquet(candidates_parquet)
        features_df = pd.read_parquet(features_parquet)
        
        # Join tables on candidate_id
        combined_df = candidates_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_f"))
        logger.info(f"Loaded and joined data for {len(combined_df):,} candidates.")
        
        # Calculate behavioral strength for each candidate
        strengths = []
        for row in combined_df.itertuples():
            # 1. Open to work flag (20%)
            open_to_work = 1.0 if getattr(row, "open_to_work_flag", False) else 0.0
            
            # 2. Response rate (20%)
            response_rate = float(getattr(row, "signal_recruiter_response_rate", 0.0) or 0.0)
            
            # 3. Activity recency (15%)
            last_active_s = getattr(row, "signal_last_active_date", None)
            days_inactive = 180
            if last_active_s:
                try:
                    active_date = datetime.strptime(last_active_s[:10], "%Y-%m-%d")
                    days_inactive = (reference_date - active_date).days
                except Exception:
                    pass
            recency = max(0.0, 1.0 - days_inactive / 180.0)
            
            # 4. Interview completion rate (15%)
            interview_completion = float(getattr(row, "signal_interview_completion_rate", 0.0) or 0.0)
            
            # 5. Offer acceptance rate (10%)
            offer_acceptance = float(getattr(row, "signal_offer_acceptance_rate", 0.0) or 0.0)
            if offer_acceptance < 0:
                offer_acceptance = 0.0
                
            # 6. Profile views (10%)
            views = min(float(getattr(row, "signal_profile_views_received_30d", 0.0) or 0.0) / 50.0, 1.0)
            
            # 7. Recruiter saves (10%)
            saves = min(float(getattr(row, "signal_saved_by_recruiters_30d", 0.0) or 0.0) / 10.0, 1.0)
            
            # Combined score
            strength = (
                0.20 * open_to_work +
                0.20 * response_rate +
                0.15 * recency +
                0.15 * interview_completion +
                0.10 * offer_acceptance +
                0.10 * views +
                0.10 * saves
            )
            strengths.append(strength)
            
        combined_df["behavioral_strength"] = strengths
        
        # Construct group key using name, headline, summary, and skills
        # Clean casing and whitespace to be robust
        combined_df["group_key"] = (
            combined_df["anonymized_name"].fillna("").astype(str).str.strip().str.lower() + "|" +
            combined_df["headline"].fillna("").astype(str).str.strip().str.lower() + "|" +
            combined_df["summary"].fillna("").astype(str).str.strip().str.lower() + "|" +
            combined_df["skill_names_json"].fillna("").astype(str).str.strip().str.lower()
        )
        
        # Group candidates to find twins
        groups = combined_df.groupby("group_key")
        
        records = []
        
        for key, group in groups:
            is_twin = len(group) > 1
            
            if not is_twin:
                # Unique candidate
                row = group.iloc[0]
                records.append({
                    "candidate_id": row["candidate_id"],
                    "twin_score": 1.0,
                    "is_twin": False,
                    "behavioral_strength": float(row["behavioral_strength"])
                })
            else:
                # Group of duplicates
                max_strength = group["behavioral_strength"].max()
                if max_strength <= 0.0:
                    max_strength = 1e-9 # avoid division by zero
                    
                for _, row in group.iterrows():
                    strength = row["behavioral_strength"]
                    
                    # The twin with the highest behavioral strength is preserved at 1.0
                    # Others are penalized relative to their strength
                    if strength == max_strength:
                        score = 1.0
                    else:
                        # Penalize duplicate profiles by scaling them down (e.g. max score of 0.5)
                        score = float(0.5 * (strength / max_strength))
                        
                    records.append({
                        "candidate_id": row["candidate_id"],
                        "twin_score": score,
                        "is_twin": True,
                        "behavioral_strength": float(strength)
                    })
                    
        return pd.DataFrame(records)
