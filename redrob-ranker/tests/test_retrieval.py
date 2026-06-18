import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.retrieval.engine import HybridRetriever
from src.jd_understanding.parser import JDProfile

def test_hybrid_retriever_basic(tmp_path):
    candidates_df = pd.DataFrame([
        {
            "candidate_id": "C_PYTHON", 
            "normalized_text": "[HEADLINE] Python ML Engineer [SUMMARY] Building AI models using PyTorch. [SKILLS] Python, PyTorch, SQL"
        },
        {
            "candidate_id": "C_GOLANG", 
            "normalized_text": "[HEADLINE] Go Developer [SUMMARY] Backend engineer writing Go and Docker systems. [SKILLS] Go, Docker, Kubernetes"
        },
        {
            "candidate_id": "C_FRONTEND", 
            "normalized_text": "[HEADLINE] React Engineer [SUMMARY] Frontend developer writing React, TypeScript, HTML. [SKILLS] JavaScript, React, TypeScript"
        }
    ])

    jd = JDProfile(
        required_skills=["Python", "PyTorch"],
        preferred_skills=["SQL"],
        soft_skills=["Communication"],
        location_requirements=["Bengaluru"],
        industry_requirements=["SaaS"]
    )

    retriever = HybridRetriever(rrf_k=60)
    
    # We use a temporary directory for cache to test caching logic
    results_df = retriever.retrieve(
        jd_profile=jd,
        candidates_df=candidates_df,
        top_n=3,
        embeddings_cache_dir=str(tmp_path)
    )

    assert len(results_df) == 3
    assert list(results_df.columns) == [
        "candidate_id", "bm25_rank", "bm25_score", 
        "dense_rank", "dense_score", "rrf_score", "combined_rank"
    ]

    # C_PYTHON must be ranked 1st
    first_row = results_df.iloc[0]
    assert first_row["candidate_id"] == "C_PYTHON"
    assert first_row["combined_rank"] == 1
    
    # Check that cache file was generated successfully
    cache_files = list(Path(tmp_path).glob("*.npy"))
    assert len(cache_files) == 1

    # Running a second retrieval should hit the cache successfully
    results_df_cached = retriever.retrieve(
        jd_profile=jd,
        candidates_df=candidates_df,
        top_n=3,
        embeddings_cache_dir=str(tmp_path)
    )
    assert results_df_cached.iloc[0]["candidate_id"] == "C_PYTHON"

def test_hybrid_retriever_empty():
    retriever = HybridRetriever()
    results_df = retriever.retrieve(
        jd_profile=JDProfile(),
        candidates_df=pd.DataFrame(),
        top_n=10
    )
    assert len(results_df) == 0
    assert "candidate_id" in results_df.columns
