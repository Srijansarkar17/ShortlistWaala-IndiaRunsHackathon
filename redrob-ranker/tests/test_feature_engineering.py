import pytest
from datetime import date
import pandas as pd
from pathlib import Path

from src.ingestion.models import Candidate, CandidateProfile, CareerEntry, RedrobSignals
from src.feature_engineering.extractor import FeatureExtractor
from src.feature_engineering.store import FeatureStore

def test_experience_features():
    extractor = FeatureExtractor()
    c = Candidate(
        candidate_id="123",
        profile=CandidateProfile(years_of_experience=5.5),
        career_history=[
            CareerEntry(title="Software Engineer", company="TechCorp", duration_months=24, industry="Product"),
            CareerEntry(title="Senior Engineer", company="TechCorp", duration_months=12, industry="Product"),
            CareerEntry(title="Consultant", company="ConsultingInc", duration_months=24, industry="Consulting")
        ]
    )
    
    features = extractor.extract(c)
    
    assert features["total_experience"] == 5.5
    assert features["avg_tenure"] == (60 / 3) / 12.0
    assert features["promotion_count"] == 1
    assert features["product_company_ratio"] == pytest.approx(2/3)
    assert features["consulting_ratio"] == pytest.approx(1/3)
    assert features["startup_ratio"] == 0.0

def test_empty_career():
    extractor = FeatureExtractor()
    c = Candidate(candidate_id="abc")
    features = extractor.extract(c)
    
    assert features["total_experience"] == 0.0
    assert features["avg_tenure"] == 0.0
    assert features["promotion_count"] == 0
    assert features["product_company_ratio"] == 0.0
    assert features["consulting_ratio"] == 0.0
    assert features["startup_ratio"] == 0.0

def test_behavioral_signals():
    extractor = FeatureExtractor()
    signals = RedrobSignals(
        profile_completeness_score=95.0,
        signup_date="2020-05-15",
        last_active_date="2022-12-01",
        open_to_work_flag=True,
        expected_salary_range_inr_lpa={"min": 15.0, "max": 25.0},
        skill_assessment_scores={"Python": 90.0, "SQL": 80.0},
        verified_email=True,
        notice_period_days=60
    )
    c = Candidate(
        candidate_id="CAND_SIG",
        redrob_signals=signals
    )
    
    features = extractor.extract(c)
    
    # Simple scalar signals
    assert features["signal_profile_completeness_score"] == 95.0
    assert features["signal_open_to_work_flag"] is True
    assert features["signal_verified_email"] is True
    
    # Date signals converted to strings
    assert features["signal_signup_date"] == "2020-05-15"
    assert features["signal_last_active_date"] == "2022-12-01"
    
    # Stated notice period (falls back under Behavior and Notice Period)
    assert features["notice_period_days"] == 60
    
    # Flattened salary dict
    assert features["signal_expected_salary_min"] == 15.0
    assert features["signal_expected_salary_max"] == 25.0
    
    # JSON-serialized skill scores
    import json
    assert json.loads(features["signal_skill_assessment_scores_json"]) == {"Python": 90.0, "SQL": 80.0}
    assert features["signal_num_skill_assessments"] == 2
    assert features["signal_avg_skill_assessment_score"] == 85.0
    assert features["signal_max_skill_assessment_score"] == 90.0
    assert features["signal_min_skill_assessment_score"] == 80.0

def test_feature_store_export(tmp_path):
    output_parquet = tmp_path / "features.parquet"
    store = FeatureStore(output_path=str(output_parquet))
    
    signals = RedrobSignals(
        profile_completeness_score=85.0,
        signup_date="2021-01-01",
        expected_salary_range_inr_lpa={"min": 10.0, "max": 12.0},
        skill_assessment_scores={"Go": 95.0}
    )
    
    candidates = [
        Candidate(candidate_id="C1", redrob_signals=signals),
        Candidate(candidate_id="C2")
    ]
    
    path, rows_written = store.export_stream(iter(candidates))
    
    assert rows_written == 2
    assert Path(path).exists()
    
    # Read back and verify columns
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "candidate_id" in df.columns
    assert "signal_profile_completeness_score" in df.columns
    assert "signal_signup_date" in df.columns
    assert "signal_expected_salary_min" in df.columns
    assert "signal_skill_assessment_scores_json" in df.columns
    assert "signal_avg_skill_assessment_score" in df.columns
    
    # Check values
    assert df.loc[df["candidate_id"] == "C1", "signal_profile_completeness_score"].values[0] == 85.0
    assert df.loc[df["candidate_id"] == "C1", "signal_signup_date"].values[0] == "2021-01-01"
    assert df.loc[df["candidate_id"] == "C1", "signal_expected_salary_min"].values[0] == 10.0
    
    import json
    scores_dict = json.loads(df.loc[df["candidate_id"] == "C1", "signal_skill_assessment_scores_json"].values[0])
    assert scores_dict == {"Go": 95.0}
    assert df.loc[df["candidate_id"] == "C1", "signal_avg_skill_assessment_score"].values[0] == 95.0
    
    # Check defaults for empty signals candidate
    assert pd.isna(df.loc[df["candidate_id"] == "C2", "signal_profile_completeness_score"].values[0])


