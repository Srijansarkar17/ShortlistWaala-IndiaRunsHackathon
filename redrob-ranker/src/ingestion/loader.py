"""
CandidateLoader
===============
Streams candidates.jsonl line-by-line, validates each record against the
Pydantic `Candidate` model, and yields successfully parsed objects.

Features
--------
- Memory-efficient streaming (no full file load)
- Per-line ValidationError capture — bad rows are skipped, not fatal
- Returns a LoadResult summary with counts and error samples
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from loguru import logger
from pydantic import ValidationError
from tqdm import tqdm

from .models import Candidate


@dataclass
class LoadResult:
    """Summary of a completed load pass."""

    total_lines: int = 0
    parsed: int = 0
    skipped_blank: int = 0
    skipped_invalid_json: int = 0
    skipped_validation: int = 0
    validation_errors: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return self.parsed / self.total_lines

    def summary(self) -> str:
        return (
            f"Lines read: {self.total_lines} | "
            f"Parsed OK: {self.parsed} | "
            f"Blank: {self.skipped_blank} | "
            f"Bad JSON: {self.skipped_invalid_json} | "
            f"Validation errors: {self.skipped_validation} | "
            f"Success rate: {self.success_rate:.2%}"
        )


class CandidateLoader:
    """
    Streams and validates candidates from a JSONL file.

    Parameters
    ----------
    path : str | Path
        Path to candidates.jsonl
    max_error_samples : int
        Maximum number of validation errors to store in LoadResult
    show_progress : bool
        Whether to display a tqdm progress bar
    """

    def __init__(
        self,
        path: str | Path,
        max_error_samples: int = 50,
        show_progress: bool = True,
    ) -> None:
        self.path = Path(path)
        self.max_error_samples = max_error_samples
        self.show_progress = show_progress

        if not self.path.exists():
            raise FileNotFoundError(f"candidates file not found: {self.path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream(self) -> tuple[Iterator[Candidate], LoadResult]:
        """
        Return a lazy iterator of validated Candidate objects and a live
        LoadResult that is populated as iteration proceeds.

        Usage::

            iterator, result = loader.stream()
            candidates = list(iterator)   # exhausts the generator
            print(result.summary())
        """
        result = LoadResult()
        return self._generate(result), result

    def load_all(self) -> tuple[list[Candidate], LoadResult]:
        """
        Eagerly load all candidates into memory.

        Returns
        -------
        (candidates, result)
        """
        it, result = self.stream()
        candidates = list(it)
        logger.info(result.summary())
        return candidates, result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(self, result: LoadResult) -> Iterator[Candidate]:
        total_lines = self._count_lines()
        result.total_lines = total_lines

        with open(self.path, "r", encoding="utf-8") as fh:
            bar = tqdm(
                fh,
                total=total_lines,
                desc="Loading candidates",
                unit="rows",
                disable=not self.show_progress,
                dynamic_ncols=True,
            )
            for raw_line in bar:
                line = raw_line.strip()

                # 1. blank lines
                if not line:
                    result.skipped_blank += 1
                    continue

                # 2. JSON decode
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    result.skipped_invalid_json += 1
                    logger.debug(f"Bad JSON: {exc}")
                    continue

                # 3. Pydantic validation
                try:
                    candidate = Candidate.model_validate(data)
                    result.parsed += 1
                    yield candidate
                except ValidationError as exc:
                    result.skipped_validation += 1
                    cid = data.get("candidate_id", "<unknown>")
                    if len(result.validation_errors) < self.max_error_samples:
                        result.validation_errors.append(
                            {"candidate_id": cid, "errors": exc.errors()}
                        )
                    logger.debug(f"Validation failed for {cid}: {exc}")

    def _count_lines(self) -> int:
        """Fast line count without reading full content."""
        count = 0
        with open(self.path, "rb") as fh:
            for _ in fh:
                count += 1
        return count
