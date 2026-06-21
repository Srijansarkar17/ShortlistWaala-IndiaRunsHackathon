from __future__ import annotations

from pathlib import Path
import pandas as pd
from loguru import logger

class SubmissionValidator:
    """Validates the final submission CSV file format and constraints."""

    def validate(self, csv_path: str | Path, candidates_parquet: str | Path = None) -> bool:
        """
        Runs format validation checks on the generated CSV file.
        """
        p = Path(csv_path)
        if not p.exists():
            logger.error(f"Submission file does not exist: {p}")
            return False

        # 1. Check UTF-8 Encoding
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"File '{p.name}' is valid UTF-8.")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding check failed: File is not valid UTF-8: {e}")
            return False

        # 2. Load DataFrame
        try:
            df = pd.read_csv(p)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False

        # 3. Column Structure
        expected_cols = ["candidate_id", "rank", "score", "reasoning"]
        if list(df.columns) != expected_cols:
            logger.error(f"Columns mismatch! Expected: {expected_cols}, got: {list(df.columns)}")
            return False

        # 4. Exactly 100 rows
        n_rows = len(df)
        if n_rows != 100:
            logger.error(f"Row count mismatch! Expected exactly 100 rows, got {n_rows}")
            return False

        # 5. Ranks 1 to 100 unique
        ranks = df["rank"].tolist()
        if sorted(ranks) != list(range(1, 101)):
            logger.error("Ranks must use each integer from 1 through 100 exactly once.")
            return False

        # 6. Candidate IDs unique
        cids = df["candidate_id"].tolist()
        if len(set(cids)) != 100:
            logger.error("Duplicate candidate_ids found in submission.")
            return False

        # 7. Candidate IDs must exist in candidates database
        if candidates_parquet:
            df_cand = pd.read_parquet(candidates_parquet)
            valid_cids = set(df_cand["candidate_id"])
            invalid = [cid for cid in cids if cid not in valid_cids]
            if invalid:
                logger.error(f"Found candidate_ids not present in candidates.parquet: {invalid}")
                return False

        # 8. Scores are monotonic non-increasing
        scores = df["score"].tolist()
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                logger.error(f"Scores are not monotonic! Row {i} (score: {scores[i]}) is less than Row {i+1} (score: {scores[i+1]})")
                return False

        # 9. Reasoning length & completeness
        for idx, row in df.iterrows():
            reason = str(row["reasoning"])
            if not reason or reason.strip() == "nan" or len(reason.strip()) < 5:
                logger.error(f"Missing or invalid reasoning at rank {row['rank']} for candidate {row['candidate_id']}")
                return False

        logger.info("Submission validation completed successfully! All checks passed.")
        return True
