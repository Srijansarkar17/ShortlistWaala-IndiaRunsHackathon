from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pytest
import pandas as pd

from src.validation.twin_resolver import TwinResolver

def test_twin_resolver_scoring_and_grouping(tmp_path):
    # 1. Create a dummy candidates database with twin and unique profiles
    # Grouping is based on: anonymized_name, headline, summary, skill_names_json
    candidates_df = pd.DataFrame([
        # Unique candidate 1
        {
            "candidate_id": "C_UNIQUE_1",
            "anonymized_name": "John Doe",
            "headline": "Software Engineer",
            "summary": "Full stack developer.",
            "skill_names_json": json.dumps(["Python", "React"]),
            "open_to_work_flag": True,
        },
        # Unique candidate 2
        {
            "candidate_id": "C_UNIQUE_2",
            "anonymized_name": "Jane Smith",
            "headline": "Data Scientist",
            "summary": "Machine learning specialist.",
            "skill_names_json": json.dumps(["Python", "PyTorch"]),
            "open_to_work_flag": False,
        },
        # Twin group 1 - Candidate A (Strongest behavioral signals)
        {
            "candidate_id": "C_TWIN_1_A",
            "anonymized_name": "Alice Wonderland",
            "headline": "Product Manager",
            "summary": "Building great products.",
            "skill_names_json": json.dumps(["Agile", "Jira"]),
            "open_to_work_flag": True,
        },
        # Twin group 1 - Candidate B (Medium behavioral signals)
        {
            "candidate_id": "C_TWIN_1_B",
            "anonymized_name": "Alice Wonderland",
            "headline": "Product Manager",
            "summary": "Building great products.",
            "skill_names_json": json.dumps(["Agile", "Jira"]),
            "open_to_work_flag": False,
        },
        # Twin group 1 - Candidate C (Weakest behavioral signals)
        {
            "candidate_id": "C_TWIN_1_C",
            "anonymized_name": "Alice Wonderland",
            "headline": "Product Manager",
            "summary": "Building great products.",
            "skill_names_json": json.dumps(["Agile", "Jira"]),
            "open_to_work_flag": False,
        },
    ])
    candidates_path = tmp_path / "candidates.parquet"
    candidates_df.to_parquet(candidates_path, index=False)

    # 2. Create features store with matching candidate_ids
    # Features contain signals
    features_df = pd.DataFrame([
        {
            "candidate_id": "C_UNIQUE_1",
            "signal_recruiter_response_rate": 0.8,
            "signal_last_active_date": "2026-06-20",
            "signal_interview_completion_rate": 0.9,
            "signal_offer_acceptance_rate": 0.7,
            "signal_profile_views_received_30d": 25,
            "signal_saved_by_recruiters_30d": 5,
        },
        {
            "candidate_id": "C_UNIQUE_2",
            "signal_recruiter_response_rate": 0.5,
            "signal_last_active_date": "2026-06-01",
            "signal_interview_completion_rate": 0.6,
            "signal_offer_acceptance_rate": 0.5,
            "signal_profile_views_received_30d": 10,
            "signal_saved_by_recruiters_30d": 2,
        },
        # Twin 1_A: High signals
        {
            "candidate_id": "C_TWIN_1_A",
            "signal_recruiter_response_rate": 0.9,
            "signal_last_active_date": "2026-06-21",
            "signal_interview_completion_rate": 0.95,
            "signal_offer_acceptance_rate": 0.8,
            "signal_profile_views_received_30d": 40,
            "signal_saved_by_recruiters_30d": 8,
        },
        # Twin 1_B: Medium signals
        {
            "candidate_id": "C_TWIN_1_B",
            "signal_recruiter_response_rate": 0.4,
            "signal_last_active_date": "2026-05-15",
            "signal_interview_completion_rate": 0.5,
            "signal_offer_acceptance_rate": 0.3,
            "signal_profile_views_received_30d": 5,
            "signal_saved_by_recruiters_30d": 1,
        },
        # Twin 1_C: Low signals
        {
            "candidate_id": "C_TWIN_1_C",
            "signal_recruiter_response_rate": 0.1,
            "signal_last_active_date": "2026-01-01",
            "signal_interview_completion_rate": 0.1,
            "signal_offer_acceptance_rate": 0.1,
            "signal_profile_views_received_30d": 0,
            "signal_saved_by_recruiters_30d": 0,
        },
    ])
    features_path = tmp_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    # 3. Resolve twins
    resolver = TwinResolver()
    ref_date = datetime(2026, 6, 21)
    
    twin_df = resolver.resolve(
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        reference_date=ref_date
    )

    # 4. Assertions
    assert len(twin_df) == 5
    assert set(twin_df.columns) == {"candidate_id", "twin_score", "is_twin", "behavioral_strength"}

    # Unique candidates should not be twins and should have twin_score = 1.0
    u1 = twin_df[twin_df["candidate_id"] == "C_UNIQUE_1"].iloc[0]
    assert not bool(u1["is_twin"])
    assert u1["twin_score"] == 1.0

    u2 = twin_df[twin_df["candidate_id"] == "C_UNIQUE_2"].iloc[0]
    assert not bool(u2["is_twin"])
    assert u2["twin_score"] == 1.0

    # Twin candidates should have is_twin = True
    t1a = twin_df[twin_df["candidate_id"] == "C_TWIN_1_A"].iloc[0]
    t1b = twin_df[twin_df["candidate_id"] == "C_TWIN_1_B"].iloc[0]
    t1c = twin_df[twin_df["candidate_id"] == "C_TWIN_1_C"].iloc[0]

    assert bool(t1a["is_twin"])
    assert bool(t1b["is_twin"])
    assert bool(t1c["is_twin"])

    # Candidate A has the highest behavioral strength in the twin group, so its score must be 1.0
    assert t1a["behavioral_strength"] > t1b["behavioral_strength"]
    assert t1b["behavioral_strength"] > t1c["behavioral_strength"]

    assert t1a["twin_score"] == 1.0
    # Candidate B and C should have twin_score < 1.0
    assert t1b["twin_score"] < 1.0
    assert t1c["twin_score"] < 1.0
    # Score for non-best twins should be 0.5 * (strength / max_strength)
    expected_b_score = 0.5 * (t1b["behavioral_strength"] / t1a["behavioral_strength"])
    assert pytest.approx(t1b["twin_score"]) == expected_b_score
