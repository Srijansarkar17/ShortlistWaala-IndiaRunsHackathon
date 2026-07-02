from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.ranking.engine import RankingEngine, ALL_FEATURES

def test_ranking_engine_pipeline(tmp_path):
    # 1. Create mock dataframes for all phases
    candidate_ids = [f"CAND_{i:03d}" for i in range(100)]
    
    # Candidates
    df_candidates = pd.DataFrame({
        "candidate_id": candidate_ids,
        "anonymized_name": [f"Name {i}" for i in range(100)],
        "headline": ["Software Engineer"] * 100,
        "summary": ["Detailed summary"] * 100,
        "skill_names_json": [json.dumps(["Python"])] * 100,
        "years_of_experience": np.random.uniform(1.0, 15.0, 100),
    })
    candidates_path = tmp_path / "candidates.parquet"
    df_candidates.to_parquet(candidates_path, index=False)

    # Features (Behavioral)
    df_features = pd.DataFrame({
        "candidate_id": candidate_ids,
        "total_experience": np.random.uniform(1.0, 15.0, 100),
        "relevant_experience": np.random.uniform(0.5, 12.0, 100),
        "avg_tenure": np.random.uniform(1.0, 5.0, 100),
        "promotion_count": np.random.randint(0, 4, 100),
        "product_company_ratio": np.random.uniform(0.0, 1.0, 100),
        "consulting_ratio": np.random.uniform(0.0, 1.0, 100),
        "startup_ratio": np.random.uniform(0.0, 1.0, 100),
        "signal_profile_completeness_score": np.random.uniform(50.0, 100.0, 100),
        "signal_recruiter_response_rate": np.random.uniform(0.0, 1.0, 100),
        "signal_interview_completion_rate": np.random.uniform(0.0, 1.0, 100),
        "signal_offer_acceptance_rate": np.random.uniform(0.0, 1.0, 100),
        "signal_profile_views_received_30d": np.random.randint(0, 100, 100),
        "signal_saved_by_recruiters_30d": np.random.randint(0, 20, 100),
        "signal_connection_count": np.random.randint(0, 500, 100),
        "signal_github_activity_score": np.random.uniform(0.0, 10.0, 100),
        "signal_notice_period_days": np.random.randint(0, 90, 100),
    })
    features_path = tmp_path / "features.parquet"
    df_features.to_parquet(features_path, index=False)

    # Honeypots
    df_honeypots = pd.DataFrame({
        "candidate_id": candidate_ids,
        "honeypot_score": [0.1] * 95 + [0.9] * 5, # 5 honeypots
        "is_honeypot": [False] * 95 + [True] * 5,
    })
    honeypots_path = tmp_path / "honeypots.parquet"
    df_honeypots.to_parquet(honeypots_path, index=False)

    # Retrieval
    df_retrieval = pd.DataFrame({
        "candidate_id": candidate_ids,
        "bm25_score": np.random.uniform(1.0, 20.0, 100),
        "dense_score": np.random.uniform(0.3, 0.9, 100),
        "rrf_score": np.random.uniform(0.01, 0.1, 100),
        "bm25_rank": np.arange(1, 101),
        "dense_rank": np.arange(1, 101),
    })
    retrieval_path = tmp_path / "retrieval_results.parquet"
    df_retrieval.to_parquet(retrieval_path, index=False)

    # JD Features
    df_jd = pd.DataFrame({
        "candidate_id": candidate_ids,
        "feat_required_skills_match_ratio": np.random.uniform(0.0, 1.0, 100),
        "feat_preferred_skills_match_ratio": np.random.uniform(0.0, 1.0, 100),
        "feat_skills_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_experience_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_is_technical_role": [1.0] * 100,
        "feat_is_ai_ml_role": [1.0] * 100,
        "feat_location_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_domain_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_responsibility_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_certification_match_score": np.random.uniform(0.0, 1.0, 100),
        "feat_semantic_match": np.random.uniform(0.0, 1.0, 100),
    })
    jd_features_path = tmp_path / "jd_features.parquet"
    df_jd.to_parquet(jd_features_path, index=False)

    # Trap Features
    df_trap = pd.DataFrame({
        "candidate_id": candidate_ids,
        "trap_consulting_only": np.random.uniform(0.0, 1.0, 100),
        "trap_research_only": np.random.uniform(0.0, 1.0, 100),
        "trap_langchain_only": np.random.uniform(0.0, 1.0, 100),
        "trap_architect_no_code": np.random.uniform(0.0, 1.0, 100),
        "trap_job_hopper": np.random.uniform(0.0, 1.0, 100),
        "trap_keyword_stuffer": np.random.uniform(0.0, 1.0, 100),
        "trap_marketing_summary": np.random.uniform(0.0, 1.0, 100),
        "trap_inactive_candidate": np.random.uniform(0.0, 1.0, 100),
        "trap_cv_only": np.random.uniform(0.0, 1.0, 100),
        "trap_robotics_only": np.random.uniform(0.0, 1.0, 100),
        "trap_speech_only": np.random.uniform(0.0, 1.0, 100),
    })
    trap_features_path = tmp_path / "trap_features.parquet"
    df_trap.to_parquet(trap_features_path, index=False)

    # Twin Features
    df_twin = pd.DataFrame({
        "candidate_id": candidate_ids,
        "twin_score": [1.0] * 90 + [0.5] * 10,
        "behavioral_strength": np.random.uniform(0.1, 0.9, 100),
    })
    twin_features_path = tmp_path / "twin_features.parquet"
    df_twin.to_parquet(twin_features_path, index=False)

    # 2. Run compile_dataset
    engine = RankingEngine()
    df_compiled = engine.compile_dataset(
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        honeypots_parquet=honeypots_path,
        retrieval_parquet=retrieval_path,
        jd_features_parquet=jd_features_path,
        trap_features_parquet=trap_features_path,
        twin_features_parquet=twin_features_path,
    )

    assert df_compiled.shape[0] == 100
    # Make sure all features exist
    for f in ALL_FEATURES:
        assert f in df_compiled.columns

    # 3. Test building training groups
    X, y, groups = engine.build_training_groups(df_compiled)
    assert X.shape[0] == 500  # 5 groups * 100 candidates
    assert X.shape[1] == len(ALL_FEATURES)
    assert len(y) == 500
    assert groups == [100, 100, 100, 100, 100]

    # Verify that honeypot candidates have relevance label = 0
    df_compiled_reset = df_compiled.reset_index(drop=True)
    honeypot_indices = df_compiled_reset[df_compiled_reset["is_honeypot"] == True].index.tolist()
    for g_idx in range(5):
        for idx in honeypot_indices:
            label_idx = g_idx * 100 + idx
            assert y.iloc[label_idx] == 0

    # 4. Train the LightGBM model
    model_path = tmp_path / "lambdarank.txt"
    ranker = engine.train(df_compiled, model_output_path=model_path)
    assert model_path.exists()

    # 5. Run inference
    ranked_df = engine.rank(df_compiled, model_path=model_path)
    assert len(ranked_df) == 100
    assert list(ranked_df.columns) == ["candidate_id", "score", "rank"]
    
    # Verify scores are strictly non-increasing
    scores = ranked_df["score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

    # Verify rank column corresponds to indices + 1
    assert (ranked_df["rank"] == np.arange(1, 101)).all()
