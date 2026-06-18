import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.jd_understanding.parser import JDProfile

class HybridRetriever:
    """lexical (BM25) and semantic (Dense) hybrid retrieval engine using RRF."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", rrf_k: int = 60):
        self.model_name = model_name
        self.rrf_k = rrf_k
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading dense embedding model: {self.model_name} on CPU")
            # Force CPU processing for compatibility and consistency
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def retrieve(
        self,
        jd_profile: JDProfile,
        candidates_df: pd.DataFrame,
        top_n: int = 5000,
        embeddings_cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieves top_n candidates using a combination of BM25 and Dense Embeddings.
        
        Parameters
        ----------
        jd_profile : JDProfile
            The Job Description parsed profile.
        candidates_df : pd.DataFrame
            DataFrame containing candidates with 'candidate_id' and 'normalized_text'.
        top_n : int
            Number of top candidates to retrieve (default: 5000).
        embeddings_cache_dir : Optional[str]
            Path to folder containing candidates embeddings cache. If provided,
            dense embeddings will be cached to avoid recomputation on subsequent runs.
        """
        if candidates_df.empty:
            logger.warning("Candidates DataFrame is empty. Returning empty results.")
            return pd.DataFrame(columns=[
                "candidate_id", "bm25_rank", "bm25_score", 
                "dense_rank", "dense_score", "rrf_score", "combined_rank"
            ])

        # 1. Build Query from JDProfile
        query_text = self._build_query_string(jd_profile)
        logger.info(f"Constructed search query: '{query_text}'")

        # 2. Lexical Retrieval: BM25
        logger.info("Computing lexical (BM25) rankings...")
        bm25_scores = self._compute_bm25_scores(candidates_df, query_text)
        
        # Create BM25 rank mappings (1-indexed)
        # Using method='first' to resolve ties deterministically
        bm25_ranks = pd.Series(bm25_scores).rank(ascending=False, method="first").astype(int).tolist()

        # 3. Semantic Retrieval: Dense Embeddings
        logger.info("Computing semantic (Dense) rankings...")
        dense_scores = self._compute_dense_scores(candidates_df, query_text, embeddings_cache_dir)
        
        # Create Dense rank mappings (1-indexed)
        dense_ranks = pd.Series(dense_scores).rank(ascending=False, method="first").astype(int).tolist()

        # 4. Reciprocal Rank Fusion (RRF)
        logger.info("Applying Reciprocal Rank Fusion (RRF)...")
        
        # Build RRF scores and collect data
        rrf_scores = []
        for i in range(len(candidates_df)):
            r_bm25 = bm25_ranks[i]
            r_dense = dense_ranks[i]
            
            # RRF Formula
            score = (1.0 / (self.rrf_k + r_bm25)) + (1.0 / (self.rrf_k + r_dense))
            rrf_scores.append(score)

        # 5. Compile and sort results
        candidate_ids = candidates_df["candidate_id"].tolist()
        results_df = pd.DataFrame({
            "candidate_id": candidate_ids,
            "bm25_rank": bm25_ranks,
            "bm25_score": bm25_scores,
            "dense_rank": dense_ranks,
            "dense_score": dense_scores,
            "rrf_score": rrf_scores
        })

        # Sort by rrf_score descending, and break ties with dense_score, then candidate_id
        results_df = results_df.sort_values(
            by=["rrf_score", "dense_score", "candidate_id"], 
            ascending=[False, False, True]
        ).reset_index(drop=True)

        # Assign combined rank (1-indexed)
        results_df["combined_rank"] = results_df.index + 1

        # Keep top N
        top_results = results_df.head(top_n).copy()
        logger.info(f"Successfully retrieved top {len(top_results)} candidates.")
        
        return top_results

    def _build_query_string(self, jd_profile: JDProfile) -> str:
        """Combine skills, industries, and soft skills into a clean search query."""
        parts = []
        if jd_profile.required_skills:
            parts.extend(jd_profile.required_skills)
        if jd_profile.preferred_skills:
            parts.extend(jd_profile.preferred_skills)
        if jd_profile.soft_skills:
            parts.extend(jd_profile.soft_skills)
        if jd_profile.industry_requirements:
            parts.extend(jd_profile.industry_requirements)
            
        # Fallback to general terms if empty
        if not parts:
            return "Software Engineer"
            
        return " ".join(parts)

    def _compute_bm25_scores(self, df: pd.DataFrame, query: str) -> np.ndarray:
        """Compute BM25 match scores for each document."""
        # Simple split tokenizer
        corpus = [doc.lower().split() for doc in df["normalized_text"].fillna("").tolist()]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        return np.array(bm25.get_scores(query_tokens))

    def _compute_dense_scores(
        self,
        df: pd.DataFrame,
        query: str,
        cache_dir: Optional[str]
    ) -> np.ndarray:
        """Compute cosine similarity dense scores, using numpy file cache if available."""
        texts = df["normalized_text"].fillna("").tolist()
        
        # Load or compute document embeddings
        embeddings = None
        cache_file = None
        
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            # Create a deterministic filename based on doc length and row count
            cache_file = cache_path / f"embeddings_cache_{len(df)}_docs.npy"
            
            if cache_file.exists():
                logger.info(f"Loading document embeddings from cache: {cache_file}")
                try:
                    embeddings = np.load(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to load embedding cache: {e}. Recomputing...")
                    
        if embeddings is None:
            logger.info("Generating dense embeddings for candidates...")
            # batch size 256 for memory efficiency and throughput
            embeddings = self.model.encode(
                texts,
                batch_size=256,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            if cache_file:
                logger.info(f"Saving computed embeddings to cache: {cache_file}")
                np.save(cache_file, embeddings)

        # Encode Query
        query_emb = self.model.encode(query, convert_to_numpy=True)
        
        # Memory-efficient Cosine Similarity computation via dot product on normalized vectors
        logger.info("Computing cosine similarities...")
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        doc_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normalized_docs = embeddings / doc_norms
        dense_scores = np.dot(normalized_docs, query_norm)
        
        return dense_scores
