from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from loguru import logger
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Reranks candidates using a Transformer cross-encoder model with CPU optimization and persistent caching."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            local_path = Path(__file__).resolve().parents[2] / "artifacts" / "models" / "ms-marco-MiniLM-L-6-v2"
            model_path = str(local_path) if local_path.exists() else self.model_name
            logger.info(f"Loading CrossEncoder model from {model_path} on CPU...")
            self._model = CrossEncoder(model_path, device="cpu")
        return self._model

    def rerank(
        self,
        top_n_parquet: str | Path,
        candidates_parquet: str | Path,
        jd_txt_path: str | Path,
        cache_json_path: str | Path,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Reranks the top candidate pool from the ranking engine output using the cross-encoder.
        """
        logger.info("Initializing Cross-Encoder reranking step...")

        # 1. Load job description text
        jd_path = Path(jd_txt_path)
        if not jd_path.exists():
            raise FileNotFoundError(f"Job Description file not found at: {jd_path}")
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read().strip()

        # 2. Load top candidates from ranking engine
        df_ranked = pd.read_parquet(top_n_parquet)
        candidate_ids = df_ranked["candidate_id"].tolist()
        logger.info(f"Loaded {len(candidate_ids):,} candidates to rerank.")

        # 3. Load normalized profiles
        df_candidates = pd.read_parquet(candidates_parquet)
        # Filter to only the candidates in our ranking pool
        df_candidates = df_candidates[df_candidates["candidate_id"].isin(candidate_ids)].copy()
        id_to_text = dict(zip(df_candidates["candidate_id"], df_candidates["normalized_text"]))

        # 4. Load persistent cache
        cache_path = Path(cache_json_path)
        cache = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache):,} cached scores from: {cache_path}")
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}. Starting with empty cache.")

        # 5. Determine which candidates need inference
        candidates_to_score = []
        pairs_to_score = []
        final_scores = {}

        for cid in candidate_ids:
            if cid in cache:
                final_scores[cid] = float(cache[cid])
            else:
                text = id_to_text.get(cid, "")
                candidates_to_score.append(cid)
                pairs_to_score.append([jd_text, text])

        # 6. Run batch inference on CPU for non-cached candidates
        if pairs_to_score:
            logger.info(f"Running Cross-Encoder inference for {len(pairs_to_score):,} candidates...")
            # Predict scores
            scores = self.model.predict(
                pairs_to_score,
                batch_size=self.batch_size,
                show_progress_bar=True
            )
            # Map predictions
            for cid, score in zip(candidates_to_score, scores):
                # convert numpy float to standard python float for JSON serialization
                final_scores[cid] = float(score)
                cache[cid] = float(score)

            # Update cache file
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
                logger.info(f"Saved updated cache to: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to write cache file: {e}")
        else:
            logger.info("All candidates retrieved from cache. No inference needed.")

        # 7. Construct and return final ranked DataFrame
        records = []
        for cid in candidate_ids:
            records.append({
                "candidate_id": cid,
                "score": final_scores.get(cid, -9999.0)
            })

        df_reranked = pd.DataFrame(records)
        df_reranked = df_reranked.sort_values(by="score", ascending=False).reset_index(drop=True)
        df_reranked["rank"] = df_reranked.index + 1

        logger.info(f"Selecting top {top_k} reranked candidates...")
        return df_reranked.head(top_k)
