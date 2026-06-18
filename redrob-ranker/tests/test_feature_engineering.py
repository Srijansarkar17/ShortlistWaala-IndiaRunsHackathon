import pytest

from src.ingestion.models import Candidate, CandidateProfile, CareerEntry
from src.feature_engineering.extractor import FeatureExtractor

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
