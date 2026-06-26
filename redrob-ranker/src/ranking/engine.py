from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import lightgbm as lgb

# Define input feature columns to be used by the model
RETRIEVAL_FEATURES = [
    "bm25_score", "dense_score", "rrf_score", "bm25_rank", "dense_rank"
]

JD_FEATURES = [
    "feat_required_skills_match_ratio",
    "feat_preferred_skills_match_ratio",
    "feat_skills_match_score",
    "feat_experience_match_score",
    "feat_location_match_score",
    "feat_domain_match_score",
    "feat_responsibility_match_score",
    "feat_certification_match_score",
    "feat_semantic_match",
    "feat_is_technical_role",
    "feat_is_ai_ml_role",
]

BEHAVIOR_FEATURES = [
    "total_experience",
    "relevant_experience",
    "avg_tenure",
    "promotion_count",
    "product_company_ratio",
    "consulting_ratio",
    "startup_ratio",
    "signal_profile_completeness_score",
    "signal_recruiter_response_rate",
    "signal_interview_completion_rate",
    "signal_offer_acceptance_rate",
    "signal_profile_views_received_30d",
    "signal_saved_by_recruiters_30d",
    "signal_connection_count",
    "signal_github_activity_score",
    "signal_notice_period_days",
]

TRAP_FEATURES = [
    "trap_consulting_only",
    "trap_research_only",
    "trap_langchain_only",
    "trap_architect_no_code",
    "trap_job_hopper",
    "trap_keyword_stuffer",
    "trap_marketing_summary",
    "trap_inactive_candidate",
    "trap_cv_only",
    "trap_robotics_only",
    "trap_speech_only",
]

TWIN_FEATURES = [
    "twin_score",
    "behavioral_strength",
]

HONEYPOT_FEATURES = [
    "honeypot_score",
]

ALL_FEATURES = (
    RETRIEVAL_FEATURES +
    JD_FEATURES +
    BEHAVIOR_FEATURES +
    TRAP_FEATURES +
    TWIN_FEATURES +
    HONEYPOT_FEATURES
)

class RankingEngine:
    """Compiles ranking features, trains a LightGBM LambdaRank model, and runs inference."""

    def compile_dataset(
        self,
        candidates_parquet: str | Path,
        features_parquet: str | Path,
        honeypots_parquet: str | Path,
        retrieval_parquet: str | Path,
        jd_features_parquet: str | Path,
        trap_features_parquet: str | Path,
        twin_features_parquet: str | Path,
    ) -> pd.DataFrame:
        """
        Loads and joins all feature dataframes on candidate_id.
        """
        logger.info("Compiling datasets for ranking model...")

        df_candidates = pd.read_parquet(candidates_parquet)
        df_features = pd.read_parquet(features_parquet)
        df_honeypots = pd.read_parquet(honeypots_parquet)
        df_retrieval = pd.read_parquet(retrieval_parquet)
        df_jd_features = pd.read_parquet(jd_features_parquet)
        df_trap_features = pd.read_parquet(trap_features_parquet)
        df_twin_features = pd.read_parquet(twin_features_parquet)

        # Merge iteratively
        merged_df = df_candidates.copy()
        
        # We perform inner/left joins on candidate_id
        for df, name in [
            (df_retrieval, "retrieval"),
            (df_jd_features, "jd_features"),
            (df_features, "features"),
            (df_trap_features, "trap_features"),
            (df_twin_features, "twin_features"),
            (df_honeypots, "honeypots"),
        ]:
            # Exclude overlapping column names except candidate_id
            cols_to_use = [c for c in df.columns if c == "candidate_id" or c not in merged_df.columns]
            merged_df = merged_df.merge(df[cols_to_use], on="candidate_id", how="inner")
            logger.info(f"Joined {name} data, current shape: {merged_df.shape}")

        return merged_df

    def build_training_groups(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[int]]:
        """
        Generates 5 synthetic query groups by evaluating the candidate pool
        under 5 distinct recruiter personas/weight schemes.
        """
        logger.info("Generating training data across 5 recruiter personas...")

        personas = [
            # Persona 0: Balanced (Default)
            {
                "weights": {"skills": 3.0, "experience": 2.0, "semantic": 2.0, "responsibility": 1.5, "location": 1.0, "domain": 1.0, "certification": 0.5, "behavior": 1.5},
                "traps": {c: 1.5 for c in TRAP_FEATURES}
            },
            # Persona 1: Tech & Skill Heavy
            {
                "weights": {"skills": 5.0, "experience": 1.0, "semantic": 3.5, "responsibility": 2.0, "location": 0.5, "domain": 0.5, "certification": 0.5, "behavior": 2.0},
                "traps": {"trap_keyword_stuffer": 3.0, "trap_cv_only": 2.0, **{c: 1.0 for c in TRAP_FEATURES if c not in ["trap_keyword_stuffer", "trap_cv_only"]}}
            },
            # Persona 2: Experience & Domain Heavy
            {
                "weights": {"skills": 1.5, "experience": 4.5, "semantic": 1.5, "responsibility": 1.0, "location": 1.0, "domain": 3.0, "certification": 1.0, "behavior": 1.0},
                "traps": {"trap_job_hopper": 3.0, "trap_consulting_only": 2.0, **{c: 1.0 for c in TRAP_FEATURES if c not in ["trap_job_hopper", "trap_consulting_only"]}}
            },
            # Persona 3: Urgent & High Engagement
            {
                "weights": {"skills": 2.0, "experience": 1.5, "semantic": 1.5, "responsibility": 1.0, "location": 1.0, "domain": 1.0, "certification": 0.5, "behavior": 4.5},
                "traps": {"trap_inactive_candidate": 3.5, "trap_cv_only": 3.0, **{c: 1.0 for c in TRAP_FEATURES if c not in ["trap_inactive_candidate", "trap_cv_only"]}}
            },
            # Persona 4: Risk-Averse (Trap / Honeypot sensitive)
            {
                "weights": {"skills": 3.0, "experience": 2.0, "semantic": 2.0, "responsibility": 1.5, "location": 1.0, "domain": 1.0, "certification": 0.5, "behavior": 1.5},
                "traps": {c: 3.5 for c in TRAP_FEATURES}
            }
        ]

        X_list = []
        y_list = []
        group_sizes = []

        n_candidates = len(df)

        for p_idx, persona in enumerate(personas):
            weights = persona["weights"]
            traps = persona["traps"]

            # Compute continuous utility scores
            suitability = (
                weights["skills"] * df["feat_skills_match_score"].fillna(0.0) +
                weights["experience"] * df["feat_experience_match_score"].fillna(0.0) +
                weights["semantic"] * df["feat_semantic_match"].fillna(0.0) +
                weights["responsibility"] * df["feat_responsibility_match_score"].fillna(0.0) +
                weights["location"] * df["feat_location_match_score"].fillna(0.0) +
                weights["domain"] * df["feat_domain_match_score"].fillna(0.0) +
                weights["certification"] * df["feat_certification_match_score"].fillna(0.0)
            )

            # Apply strict experience range penalization
            yoe = df["years_of_experience"].fillna(0.0)
            exp_penalty = pd.Series(0.0, index=df.index)
            # Below min YoE (5.0)
            under_exp_mask = yoe < 5.0
            exp_penalty.loc[under_exp_mask] += (5.0 - yoe.loc[under_exp_mask]) * 4.0
            # Above max YoE (9.0)
            over_exp_mask = yoe > 9.0
            exp_penalty.loc[over_exp_mask] += (yoe.loc[over_exp_mask] - 9.0) * 3.0
            
            # Apply strict non-technical role penalization and AI/ML role boost
            role_penalty = pd.Series(0.0, index=df.index)
            non_tech_mask = df["feat_is_technical_role"].fillna(1.0) == 0.0
            role_penalty.loc[non_tech_mask] += 30.0

            role_boost = pd.Series(0.0, index=df.index)
            ai_ml_mask = df["feat_is_ai_ml_role"].fillna(0.0) == 1.0
            role_boost.loc[ai_ml_mask] += 15.0

            behavior = weights["behavior"] * df["behavioral_strength"].fillna(0.0)

            trap_penalty = pd.Series(0.0, index=df.index)
            for trap_col, p_val in traps.items():
                trap_penalty += df[trap_col].fillna(0.0) * p_val

            # Compute utility score
            scores = suitability + behavior + role_boost - trap_penalty - exp_penalty - role_penalty
            
            # Penalize twins
            scores = scores * df["twin_score"].fillna(1.0)

            # Bin continuous utility scores into discrete relevance levels 0, 1, 2, 3, 4
            p20 = scores.quantile(0.2)
            p40 = scores.quantile(0.4)
            p60 = scores.quantile(0.6)
            p85 = scores.quantile(0.85)

            relevance = []
            for s, row in zip(scores, df.itertuples()):
                # Honeypot candidates are always tier 0 relevance
                h_score = getattr(row, "honeypot_score", 0.0)
                is_h = getattr(row, "is_honeypot", False)
                if h_score > 0.6 or is_h:
                    relevance.append(0)
                elif s <= p20:
                    relevance.append(0)
                elif s <= p40:
                    relevance.append(1)
                elif s <= p60:
                    relevance.append(2)
                elif s <= p85:
                    relevance.append(3)
                else:
                    relevance.append(4)

            # Construct feature matrix for this persona
            X_p = df[ALL_FEATURES].copy()
            # Convert boolean columns to floats to avoid any training warnings
            for col in X_p.columns:
                if X_p[col].dtype == bool:
                    X_p[col] = X_p[col].astype(float)
                else:
                    X_p[col] = X_p[col].fillna(0.0)

            X_list.append(X_p)
            y_list.extend(relevance)
            group_sizes.append(n_candidates)

        X_all = pd.concat(X_list, axis=0, ignore_index=True)
        y_all = pd.Series(y_list)

        return X_all, y_all, group_sizes

    def train(
        self,
        df_compiled: pd.DataFrame,
        model_output_path: str | Path
    ) -> lgb.LGBMRanker:
        """
        Trains the LambdaRank model and saves it to file.
        """
        logger.info("Preparing data for LightGBM LambdaRank training...")
        X, y, groups = self.build_training_groups(df_compiled)

        logger.info(f"Training set sizes: features={X.shape}, labels={len(y)}, groups={groups}")

        # Train a pairwise LambdaRank model
        # We use a moderate learning rate and shallow trees to avoid overfitting on simulated data
        # n_jobs=1 and num_threads=1 are critical on macOS to prevent OpenMP/MKL thread-pool deadlocks
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10, 50],
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            n_jobs=1,
            num_threads=1,
            verbose=-1,
        )

        logger.info("Fitting LightGBM LambdaRank model...")
        ranker.fit(
            X,
            y,
            group=groups
        )

        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        ranker.booster_.save_model(str(model_path))
        logger.info(f"Model saved successfully to: {model_path}")
        
        return ranker

    def rank(
        self,
        df_compiled: pd.DataFrame,
        model_path: str | Path
    ) -> pd.DataFrame:
        """
        Loads the saved LambdaRank model and scores candidates.
        Returns a sorted DataFrame containing candidate_id, score, rank, and features.
        """
        logger.info(f"Loading LambdaRank model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        booster = lgb.Booster(model_file=str(model_path))
        
        # Prepare inference features
        X_infer = df_compiled[ALL_FEATURES].copy()
        for col in X_infer.columns:
            if X_infer[col].dtype == bool:
                X_infer[col] = X_infer[col].astype(float)
            else:
                X_infer[col] = X_infer[col].fillna(0.0)

        # Predict scores
        logger.info("Computing ranking scores for candidates...")
        scores = booster.predict(X_infer)

        results_df = df_compiled[["candidate_id"]].copy()
        results_df["score"] = scores
        
        # Sort best first
        results_df = results_df.sort_values(by="score", ascending=False).reset_index(drop=True)
        results_df["rank"] = results_df.index + 1

        return results_df
