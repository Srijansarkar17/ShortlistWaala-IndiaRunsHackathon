from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pytest
import pandas as pd

from src.validation.trap_detector import TrapDetector

def test_trap_detector_heuristics(tmp_path):
    # 1. Create a dummy candidates database with specific trap profiles
    candidates_df = pd.DataFrame([
        {
            "candidate_id": "C_CONSULTING",
            "summary": "Experienced engineer.",
            "current_title": "Software Engineer",
            "years_of_experience": 5.0,
            "skill_names_json": json.dumps(["Python", "Java"]),
            "certification_names_json": json.dumps([]),
            "career_companies_json": json.dumps(["TCS", "Infosys", "Wipro"]),
            "career_titles_json": json.dumps(["Developer", "Software Engineer", "Systems Analyst"]),
            "career_descriptions_json": json.dumps(["Worked at TCS", "Worked at Infosys", "Worked at Wipro"]),
        },
        {
            "candidate_id": "C_RESEARCH",
            "summary": "Focus on AI research.",
            "current_title": "Postdoctoral Researcher",
            "years_of_experience": 4.0,
            "skill_names_json": json.dumps(["PyTorch", "LaTeX"]),
            "certification_names_json": json.dumps([]),
            "career_companies_json": json.dumps(["Academic Lab", "Stanford University"]),
            "career_titles_json": json.dumps(["PhD Student", "Research Fellow"]),
            "career_descriptions_json": json.dumps(["Conducted AI research", "Published papers"]),
        },
        {
            "candidate_id": "C_LANGCHAIN",
            "summary": "LangChain enthusiast building OpenAI wrappers.",
            "current_title": "AI Developer",
            "years_of_experience": 1.0,
            "skill_names_json": json.dumps(["LangChain", "OpenAI", "ChatGPT"]),
            "certification_names_json": json.dumps([]),
            "career_companies_json": json.dumps(["Startup X"]),
            "career_titles_json": json.dumps(["Intern"]),
            "career_descriptions_json": json.dumps(["Built simple ChatGPT wrappers"]),
        },
        {
            "candidate_id": "C_ROBOTICS",
            "summary": "UAV Robotics engineer.",
            "current_title": "Robotics Engineer",
            "years_of_experience": 6.0,
            "skill_names_json": json.dumps(["Robotics", "ROS", "SLAM", "C++"]),
            "certification_names_json": json.dumps([]),
            "career_companies_json": json.dumps(["RoboTech"]),
            "career_titles_json": json.dumps(["Engineer"]),
            "career_descriptions_json": json.dumps(["ROS controller design"]),
        }
    ])
    candidates_path = tmp_path / "candidates.parquet"
    candidates_df.to_parquet(candidates_path, index=False)

    # 2. Create features store
    features_df = pd.DataFrame([
        {"candidate_id": "C_CONSULTING", "num_career_entries": 3, "avg_tenure": 1.6, "profile_completeness_score": 90.0, "num_skills": 2},
        {"candidate_id": "C_RESEARCH", "num_career_entries": 2, "avg_tenure": 2.0, "profile_completeness_score": 85.0, "num_skills": 2},
        {"candidate_id": "C_LANGCHAIN", "num_career_entries": 1, "avg_tenure": 1.0, "profile_completeness_score": 75.0, "num_skills": 3},
        {"candidate_id": "C_ROBOTICS", "num_career_entries": 1, "avg_tenure": 6.0, "profile_completeness_score": 80.0, "num_skills": 4}
    ])
    features_path = tmp_path / "features.parquet"
    features_df.to_parquet(features_path, index=False)

    # 3. Detect traps
    detector = TrapDetector()
    trap_df = detector.detect(
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        reference_date=datetime(2026, 6, 21)
    )

    assert len(trap_df) == 4
    assert "candidate_id" in trap_df.columns
    assert "trap_consulting_only" in trap_df.columns
    assert "trap_research_only" in trap_df.columns
    assert "trap_langchain_only" in trap_df.columns
    assert "trap_robotics_only" in trap_df.columns

    # Verify Consulting Only
    row_consulting = trap_df[trap_df["candidate_id"] == "C_CONSULTING"].iloc[0]
    assert row_consulting["trap_consulting_only"] == 1.0
    assert row_consulting["trap_research_only"] == 0.0

    # Verify Research Only
    row_research = trap_df[trap_df["candidate_id"] == "C_RESEARCH"].iloc[0]
    assert row_research["trap_research_only"] > 0.5
    assert row_research["trap_consulting_only"] == 0.0

    # Verify LangChain Only
    row_langchain = trap_df[trap_df["candidate_id"] == "C_LANGCHAIN"].iloc[0]
    assert row_langchain["trap_langchain_only"] == 1.0

    # Verify Robotics Only
    row_robotics = trap_df[trap_df["candidate_id"] == "C_ROBOTICS"].iloc[0]
    assert row_robotics["trap_robotics_only"] == 1.0
