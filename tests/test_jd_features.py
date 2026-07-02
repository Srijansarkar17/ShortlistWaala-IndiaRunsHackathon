from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from src.jd_understanding.parser import JDProfile, ExperienceRange
from src.feature_engineering.jd_generator import JDFeatureGenerator

def test_jd_feature_generator_basic(tmp_path):
    # 1. Create a dummy candidates database
    candidates_df = pd.DataFrame([
        {
            "candidate_id": "C_MATCH",
            "years_of_experience": 6.0,
            "location": "Pune, India",
            "country": "India",
            "preferred_work_mode": "Hybrid",
            "willing_to_relocate": True,
            "current_industry": "SaaS",
            "skill_names_json": json.dumps(["Python", "PyTorch", "SQL"]),
            "certification_names_json": json.dumps(["AWS Certified Cloud Practitioner"]),
            "career_titles_json": json.dumps(["Software Engineer", "ML Engineer"]),
            "career_descriptions_json": json.dumps(["Building SaaS ranking algorithms", "PyTorch deep learning"]),
        },
        {
            "candidate_id": "C_MISMATCH",
            "years_of_experience": 2.0,
            "location": "London, UK",
            "country": "UK",
            "preferred_work_mode": "Onsite",
            "willing_to_relocate": False,
            "current_industry": "Automotive",
            "skill_names_json": json.dumps(["Java", "Spring"]),
            "certification_names_json": json.dumps([]),
            "career_titles_json": json.dumps(["Backend Java Dev"]),
            "career_descriptions_json": json.dumps(["Java Spring MVC development"]),
        }
    ])
    candidates_path = tmp_path / "candidates.parquet"
    candidates_df.to_parquet(candidates_path, index=False)

    # 2. Create precomputed candidate features
    features_df = pd.DataFrame([
        {"candidate_id": "C_MATCH", "total_experience": 6.0, "relevant_experience": 2.0},
        {"candidate_id": "C_MISMATCH", "total_experience": 2.0, "relevant_experience": 0.0}
    ])
    features_path = tmp_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    # 3. Create honeypots database
    honeypots_df = pd.DataFrame([
        {"candidate_id": "C_MATCH", "honeypot_score": 0.1, "trust_score": 0.9, "is_honeypot": False},
        {"candidate_id": "C_MISMATCH", "honeypot_score": 0.7, "trust_score": 0.3, "is_honeypot": True}
    ])
    honeypots_path = tmp_path / "honeypots.parquet"
    honeypots_df.to_parquet(honeypots_path, index=False)

    # 4. Create retrieval results
    retrieval_df = pd.DataFrame([
        {"candidate_id": "C_MATCH", "bm25_score": 12.5, "dense_score": 0.85, "rrf_score": 0.03, "combined_rank": 1}
    ])
    retrieval_path = tmp_path / "retrieval_results.parquet"
    retrieval_df.to_parquet(retrieval_path, index=False)

    # 5. Define JD Profile
    jd_profile = JDProfile(
        required_skills=["Python", "PyTorch"],
        preferred_skills=["SQL"],
        experience_range=ExperienceRange(min_years=5.0, max_years=9.0),
        location_requirements=["Pune", "Hybrid"],
        industry_requirements=["SaaS"],
        soft_skills=["Communication"]
    )

    # 6. Run feature generator without honeypot filtering
    generator = JDFeatureGenerator()
    out_df = generator.generate(
        jd_profile=jd_profile,
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        honeypots_parquet=honeypots_path,
        retrieval_parquet=retrieval_path,
        filter_honeypots=False
    )

    assert len(out_df) == 2
    assert "candidate_id" in out_df.columns
    assert "feat_skills_match_score" in out_df.columns
    
    # C_MATCH checks
    match_row = out_df[out_df["candidate_id"] == "C_MATCH"].iloc[0]
    assert match_row["feat_required_skills_match_ratio"] == 1.0
    assert match_row["feat_preferred_skills_match_ratio"] == 1.0
    assert match_row["feat_skills_match_score"] == 1.0
    assert match_row["feat_experience_match_score"] == 1.0
    assert match_row["feat_location_match_score"] == 1.0
    assert match_row["feat_domain_match_score"] == 1.0
    assert match_row["feat_retrieved"] == 1.0
    assert match_row["is_honeypot"] == False

    # C_MISMATCH checks
    mismatch_row = out_df[out_df["candidate_id"] == "C_MISMATCH"].iloc[0]
    assert mismatch_row["feat_required_skills_match_ratio"] == 0.0
    assert mismatch_row["feat_preferred_skills_match_ratio"] == 0.0
    assert mismatch_row["feat_skills_match_score"] == 0.0
    assert mismatch_row["feat_experience_match_score"] < 1.0
    assert mismatch_row["feat_location_match_score"] == 0.0
    assert mismatch_row["feat_domain_match_score"] == 0.0
    assert mismatch_row["feat_retrieved"] == 0.0
    assert mismatch_row["is_honeypot"] == True

    # 7. Run feature generator WITH honeypot filtering
    out_filtered_df = generator.generate(
        jd_profile=jd_profile,
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        honeypots_parquet=honeypots_path,
        retrieval_parquet=retrieval_path,
        filter_honeypots=True
    )
    assert len(out_filtered_df) == 1
    assert out_filtered_df.iloc[0]["candidate_id"] == "C_MATCH"
