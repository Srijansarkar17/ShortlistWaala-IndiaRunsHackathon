from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

from src.ranking.reranker import CrossEncoderReranker

def test_cross_encoder_reranker(tmp_path):
    # 1. Create dummy files
    # Dummy Job Description
    jd_path = tmp_path / "job_description.txt"
    jd_path.write_text("Looking for a Python Backend Developer.")

    # Dummy top candidates (Phase 9 output)
    ranked_path = tmp_path / "ranked_candidates.parquet"
    pd.DataFrame({
        "candidate_id": ["CAND_01", "CAND_02", "CAND_03"],
        "score": [4.5, 3.2, 2.1],
        "rank": [1, 2, 3]
    }).to_parquet(ranked_path, index=False)

    # Dummy normalized candidates
    candidates_path = tmp_path / "candidates.parquet"
    pd.DataFrame({
        "candidate_id": ["CAND_01", "CAND_02", "CAND_03"],
        "normalized_text": [
            "Python Backend Engineer with 5 years experience.",
            "Golang Developer writing microservices.",
            "React Frontend Developer building UIs."
        ]
    }).to_parquet(candidates_path, index=False)

    cache_path = tmp_path / "cross_encoder_cache.json"

    # 2. Rerank using Mocked CrossEncoder to avoid downloading the model during test
    with patch("src.ranking.reranker.CrossEncoder") as MockCrossEncoder:
        mock_model = MagicMock()
        # Mock predict to return scores corresponding to CAND_01, CAND_02, CAND_03
        mock_model.predict.return_value = [0.8, -0.4, 0.3]
        MockCrossEncoder.return_value = mock_model

        reranker = CrossEncoderReranker(batch_size=2)
        reranked_df = reranker.rerank(
            top_n_parquet=ranked_path,
            candidates_parquet=candidates_path,
            jd_txt_path=jd_path,
            cache_json_path=cache_path,
            top_k=2 # request top 2
        )

        # 3. Assertions on Reranked output
        assert len(reranked_df) == 2
        assert list(reranked_df.columns) == ["candidate_id", "score", "rank"]
        
        # Sorted order should be CAND_01 (0.8), CAND_03 (0.3), CAND_02 (-0.4)
        # Since we asked for top_k = 2, we should get CAND_01 and CAND_03
        assert reranked_df.iloc[0]["candidate_id"] == "CAND_01"
        assert reranked_df.iloc[0]["score"] == 0.8
        assert reranked_df.iloc[0]["rank"] == 1

        assert reranked_df.iloc[1]["candidate_id"] == "CAND_03"
        assert reranked_df.iloc[1]["score"] == 0.3
        assert reranked_df.iloc[1]["rank"] == 2

        # 4. Verify that cache file was saved
        assert cache_path.exists()
        with open(cache_path, "r") as f:
            saved_cache = json.load(f)
        assert saved_cache["CAND_01"] == 0.8
        assert saved_cache["CAND_02"] == -0.4
        assert saved_cache["CAND_03"] == 0.3

    # 5. Run again, using cached values (mock should NOT be called since all are cached)
    # We will modify the mock's return value to verify that cached values are preferred
    with patch("src.ranking.reranker.CrossEncoder") as MockCrossEncoder2:
        mock_model2 = MagicMock()
        mock_model2.predict.return_value = [9.9, 9.9, 9.9]
        MockCrossEncoder2.return_value = mock_model2

        reranker = CrossEncoderReranker()
        reranked_df_cached = reranker.rerank(
            top_n_parquet=ranked_path,
            candidates_parquet=candidates_path,
            jd_txt_path=jd_path,
            cache_json_path=cache_path,
            top_k=3
        )

        # Since they are loaded from cache, predict should not be called and scores must match original mock
        assert len(reranked_df_cached) == 3
        # Candidate 02 has -0.4, Candidate 03 has 0.3, Candidate 01 has 0.8
        # Sorted descending: CAND_01 (0.8), CAND_03 (0.3), CAND_02 (-0.4)
        assert reranked_df_cached.iloc[0]["candidate_id"] == "CAND_01"
        assert reranked_df_cached.iloc[0]["score"] == 0.8
        assert reranked_df_cached.iloc[1]["candidate_id"] == "CAND_03"
        assert reranked_df_cached.iloc[1]["score"] == 0.3
        assert reranked_df_cached.iloc[2]["candidate_id"] == "CAND_02"
        assert reranked_df_cached.iloc[2]["score"] == -0.4

        # mock_model2.predict should not have been called
        assert not mock_model2.predict.called
