from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from loguru import logger

from src.reasoning.generator import ReasoningGenerator
from src.ranking.engine import RankingEngine
from .validator import SubmissionValidator

class SubmissionGenerator:
    """Orchestrates submission file generation, tie breaking, reasoning generation, and format validation."""

    def generate(
        self,
        reranked_parquet: str | Path,
        candidates_parquet: str | Path,
        features_parquet: str | Path,
        honeypots_parquet: str | Path,
        retrieval_parquet: str | Path,
        jd_features_parquet: str | Path,
        trap_features_parquet: str | Path,
        twin_features_parquet: str | Path,
        output_csv_path: str | Path
    ) -> Path:
        """
        Compiles candidate features, sorts them, generates reasoning, exports to CSV, and validates.
        """
        logger.info("Generating final submission CSV...")

        # 1. Load reranked top 100 candidate IDs and scores
        df_reranked = pd.read_parquet(reranked_parquet)
        
        # 2. Compile candidate features to get matching information for reasoning
        engine = RankingEngine()
        df_compiled = engine.compile_dataset(
            candidates_parquet=candidates_parquet,
            features_parquet=features_parquet,
            honeypots_parquet=honeypots_parquet,
            retrieval_parquet=retrieval_parquet,
            jd_features_parquet=jd_features_parquet,
            trap_features_parquet=trap_features_parquet,
            twin_features_parquet=twin_features_parquet,
        )

        # 3. Join reranked scores to compiled dataset
        df_compiled = df_compiled.merge(df_reranked[["candidate_id", "score"]], on="candidate_id", how="inner")
        
        # 4. Calculate adjusted score based on experience penalties, role penalties, traps, and notice period
        adjusted_scores = []
        for row in df_compiled.itertuples():
            ce_score = row.score
            
            # --- 4.1 Experience Penalty ---
            yoe = getattr(row, "years_of_experience", 0.0)
            if pd.isna(yoe) or yoe is None:
                yoe = 0.0
            
            exp_penalty = 0.0
            if yoe < 5.0:
                exp_penalty = (5.0 - yoe) * 10.0
            elif yoe > 9.0:
                exp_penalty = (yoe - 9.0) * 8.0
                
            # --- 4.2 Technical Role Penalty & AI/ML Boost ---
            is_tech = getattr(row, "feat_is_technical_role", 1.0)
            role_penalty = 0.0
            if is_tech == 0.0:
                role_penalty = 50.0
                
            is_ai_ml = getattr(row, "feat_is_ai_ml_role", 0.0)
            role_boost = 0.0
            if is_ai_ml == 1.0:
                role_boost = 15.0
                
            # --- 4.3 Trap Penalties ---
            trap_penalties = (
                getattr(row, "trap_consulting_only", 0.0) * 10.0 +
                getattr(row, "trap_research_only", 0.0) * 10.0 +
                getattr(row, "trap_langchain_only", 0.0) * 5.0 +
                getattr(row, "trap_job_hopper", 0.0) * 8.0 +
                getattr(row, "trap_keyword_stuffer", 0.0) * 6.0 +
                getattr(row, "trap_marketing_summary", 0.0) * 20.0 +
                getattr(row, "trap_inactive_candidate", 0.0) * 10.0 +
                getattr(row, "trap_cv_only", 0.0) * 10.0 +
                getattr(row, "trap_robotics_only", 0.0) * 10.0 +
                getattr(row, "trap_speech_only", 0.0) * 10.0 +
                getattr(row, "trap_architect_no_code", 0.0) * 8.0 +
                getattr(row, "honeypot_score", 0.0) * 40.0
            )
            
            # --- 4.4 Notice Period Penalty ---
            notice_days = getattr(row, "notice_period_days", 0.0)
            if pd.isna(notice_days) or notice_days is None:
                notice_days = 0.0
            notice_penalty = 0.0
            if notice_days > 30:
                notice_penalty = (notice_days - 30) * 0.1
                
            # --- 4.5 Behavior Boost ---
            behavior_strength = getattr(row, "behavioral_strength", 0.0)
            if pd.isna(behavior_strength) or behavior_strength is None:
                behavior_strength = 0.0
            behavior_boost = behavior_strength * 2.0
            
            final_adj_score = ce_score - exp_penalty - role_penalty + role_boost - trap_penalties - notice_penalty + behavior_boost
            adjusted_scores.append(final_adj_score)
            
        df_compiled["adjusted_score"] = adjusted_scores
        # Overwrite score with adjusted_score to maintain strict monotonicity check
        df_compiled["score"] = df_compiled["adjusted_score"]
        
        # 5. Tie-breaking rule: Sort by score descending, then candidate_id ascending
        df_compiled = df_compiled.sort_values(
            by=["score", "candidate_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Slice exactly top 100
        df_compiled = df_compiled.head(100)

        # 5. Generate reasoning justifications
        reasoner = ReasoningGenerator()

        # Normalize scores to 0-100 range (min-max) so all scores are positive.
        # Sorting is already done above so rank order is preserved.
        raw_scores = df_compiled["score"].values
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        score_range = score_max - score_min if score_max != score_min else 1.0
        normalized_scores = ((raw_scores - score_min) / score_range) * 100.0

        records = []
        for idx, (row, norm_score) in enumerate(zip(df_compiled.itertuples(), normalized_scores)):
            rank = idx + 1
            reasoning = reasoner.generate_reasoning(row)
            
            records.append({
                "candidate_id": row.candidate_id,
                "rank": rank,
                "score": round(float(norm_score), 4),
                "reasoning": reasoning
            })

        df_submission = pd.DataFrame(records)


        # 6. Save as UTF-8 CSV
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_submission.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Submission CSV saved to: {out_path}")

        # 7. Run validation
        validator = SubmissionValidator()
        is_valid = validator.validate(out_path, candidates_parquet=candidates_parquet)
        if not is_valid:
            raise ValueError("Generated submission failed format validation!")

        return out_path
