from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from src.reasoning.generator import ReasoningGenerator
from src.submission.generator import SubmissionGenerator
from src.submission.validator import SubmissionValidator

def test_reasoning_generator():
    reasoner = ReasoningGenerator()
    
    # Mock candidate row using a simple object
    class DummyRow:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    row = DummyRow(
        headline="Applied ML Engineer",
        years_of_experience=6.5,
        skill_names_json=json.dumps(["Python", "PyTorch", "SQL"]),
        signal_recruiter_response_rate=0.85,
        open_to_work_flag=True,
        notice_period_days=90, # long notice period (gap)
        trap_job_hopper=0.0,
        trap_consulting_only=0.0,
        trap_inactive_candidate=0.0,
        trap_keyword_stuffer=0.0
    )

    reason = reasoner.generate_reasoning(row)
    assert "Applied ML Engineer" in reason
    # Experience must be referenced
    assert "6.5" in reason
    # Skills must be referenced
    assert "Python" in reason or "PyTorch" in reason
    # Signal must be referenced
    assert "85%" in reason
    assert "actively open" in reason
    # Gap must be referenced
    assert "90 days" in reason

def test_submission_validator(tmp_path):
    validator = SubmissionValidator()
    csv_path = tmp_path / "test_sub.csv"

    # 1. Invalid columns
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(csv_path, index=False)
    assert not validator.validate(csv_path)

    # 2. Invalid row count
    pd.DataFrame({
        "candidate_id": [f"CAND_{i}" for i in range(10)],
        "rank": list(range(1, 11)),
        "score": [1.0] * 10,
        "reasoning": ["Good fit"] * 10
    }).to_csv(csv_path, index=False)
    assert not validator.validate(csv_path)

    # 3. Duplicate ranks
    pd.DataFrame({
        "candidate_id": [f"CAND_{i}" for i in range(100)],
        "rank": [1] * 100,
        "score": [1.0] * 100,
        "reasoning": ["Good fit"] * 100
    }).to_csv(csv_path, index=False)
    assert not validator.validate(csv_path)

    # 4. Non-monotonic scores
    pd.DataFrame({
        "candidate_id": [f"CAND_{i:03d}" for i in range(100)],
        "rank": list(range(1, 101)),
        "score": list(range(100)), # increasing, not non-increasing
        "reasoning": ["Good fit"] * 100
    }).to_csv(csv_path, index=False)
    assert not validator.validate(csv_path)

def test_submission_generator_pipeline(tmp_path):
    # Prepare dummy parquet files
    candidate_ids = [f"CAND_{i:03d}" for i in range(100)]
    
    # Reranked
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "score": [float(100 - i) for i in range(100)], # monotonically decreasing
        "rank": list(range(1, 101))
    }).to_parquet(tmp_path / "reranked_candidates.parquet", index=False)

    # Candidates
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "anonymized_name": [f"Name {i}" for i in range(100)],
        "headline": ["Software Engineer"] * 100,
        "summary": ["Engineer summary"] * 100,
        "skill_names_json": [json.dumps(["Python"])] * 100,
        "years_of_experience": [6.0] * 100,
    }).to_parquet(tmp_path / "candidates.parquet", index=False)

    # Features
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "total_experience": [5.0] * 100,
        "relevant_experience": [4.0] * 100,
        "avg_tenure": [2.5] * 100,
        "promotion_count": [0] * 100,
        "product_company_ratio": [0.5] * 100,
        "consulting_ratio": [0.0] * 100,
        "startup_ratio": [0.0] * 100,
        "signal_profile_completeness_score": [80.0] * 100,
        "signal_recruiter_response_rate": [0.75] * 100,
        "signal_interview_completion_rate": [0.8] * 100,
        "signal_offer_acceptance_rate": [0.9] * 100,
        "signal_profile_views_received_30d": [10] * 100,
        "signal_saved_by_recruiters_30d": [5] * 100,
        "signal_connection_count": [100] * 100,
        "signal_github_activity_score": [5.0] * 100,
        "signal_notice_period_days": [30] * 100,
    }).to_parquet(tmp_path / "features.parquet", index=False)

    # Honeypots
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "honeypot_score": [0.1] * 100,
        "is_honeypot": [False] * 100,
    }).to_parquet(tmp_path / "honeypots.parquet", index=False)

    # Retrieval
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "bm25_score": [10.0] * 100,
        "dense_score": [0.5] * 100,
        "rrf_score": [0.02] * 100,
        "bm25_rank": list(range(1, 101)),
        "dense_rank": list(range(1, 101)),
    }).to_parquet(tmp_path / "retrieval_results.parquet", index=False)

    # JD Features
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "feat_required_skills_match_ratio": [0.8] * 100,
        "feat_preferred_skills_match_ratio": [0.5] * 100,
        "feat_skills_match_score": [0.7] * 100,
        "feat_experience_match_score": [0.8] * 100,
        "feat_is_technical_role": [1.0] * 100,
        "feat_is_ai_ml_role": [1.0] * 100,
        "feat_location_match_score": [0.9] * 100,
        "feat_domain_match_score": [0.8] * 100,
        "feat_responsibility_match_score": [0.7] * 100,
        "feat_certification_match_score": [0.6] * 100,
        "feat_semantic_match": [0.5] * 100,
    }).to_parquet(tmp_path / "jd_features.parquet", index=False)

    # Trap Features
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "trap_consulting_only": [0.0] * 100,
        "trap_research_only": [0.0] * 100,
        "trap_langchain_only": [0.0] * 100,
        "trap_architect_no_code": [0.0] * 100,
        "trap_job_hopper": [0.0] * 100,
        "trap_keyword_stuffer": [0.0] * 100,
        "trap_marketing_summary": [0.0] * 100,
        "trap_inactive_candidate": [0.0] * 100,
        "trap_cv_only": [0.0] * 100,
        "trap_robotics_only": [0.0] * 100,
        "trap_speech_only": [0.0] * 100,
    }).to_parquet(tmp_path / "trap_features.parquet", index=False)

    # Twin Features
    pd.DataFrame({
        "candidate_id": candidate_ids,
        "twin_score": [1.0] * 100,
        "behavioral_strength": [0.6] * 100,
    }).to_parquet(tmp_path / "twin_features.parquet", index=False)

    csv_out = tmp_path / "submission.csv"
    generator = SubmissionGenerator()
    final_path = generator.generate(
        reranked_parquet=tmp_path / "reranked_candidates.parquet",
        candidates_parquet=tmp_path / "candidates.parquet",
        features_parquet=tmp_path / "features.parquet",
        honeypots_parquet=tmp_path / "honeypots.parquet",
        retrieval_parquet=tmp_path / "retrieval_results.parquet",
        jd_features_parquet=tmp_path / "jd_features.parquet",
        trap_features_parquet=tmp_path / "trap_features.parquet",
        twin_features_parquet=tmp_path / "twin_features.parquet",
        output_csv_path=csv_out
    )

    assert final_path.exists()
    df_sub = pd.read_csv(final_path)
    assert len(df_sub) == 100
    assert list(df_sub.columns) == ["candidate_id", "rank", "score", "reasoning"]
