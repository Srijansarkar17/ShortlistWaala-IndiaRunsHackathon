"""
tests/test_loader.py
====================
Unit tests for CandidateLoader.

Strategy: write tiny in-memory JSONL fixtures to tmp files so tests
are fast and self-contained (no dependency on real 100k dataset).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.loader import CandidateLoader, LoadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_jsonl(lines: list, tmp_path: Path) -> Path:
    """Write a list of dicts (or raw strings) to a temp JSONL file."""
    p = tmp_path / "test_candidates.jsonl"
    with open(p, "w") as fh:
        for line in lines:
            if isinstance(line, str):
                fh.write(line + "\n")
            else:
                fh.write(json.dumps(line) + "\n")
    return p


def minimal_candidate(cid: str = "CAND_0000001") -> dict:
    return {"candidate_id": cid}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCandidateLoaderFileHandling:
    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CandidateLoader("/nonexistent/path/candidates.jsonl")

    def test_loads_valid_minimal_records(self, tmp_path):
        data = [minimal_candidate(f"CAND_{i:07d}") for i in range(5)]
        path = write_jsonl(data, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        candidates, result = loader.load_all()
        assert result.parsed == 5
        assert result.skipped_validation == 0
        assert result.skipped_blank == 0

    def test_blank_lines_are_skipped(self, tmp_path):
        lines = [
            json.dumps(minimal_candidate("CAND_0000001")),
            "",
            "   ",
            json.dumps(minimal_candidate("CAND_0000002")),
        ]
        path = write_jsonl(lines, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        candidates, result = loader.load_all()
        assert result.parsed == 2
        assert result.skipped_blank == 2

    def test_bad_json_lines_are_skipped(self, tmp_path):
        lines = [
            json.dumps(minimal_candidate("CAND_0000001")),
            "{this is not json}",
            json.dumps(minimal_candidate("CAND_0000002")),
        ]
        path = write_jsonl(lines, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        candidates, result = loader.load_all()
        assert result.parsed == 2
        assert result.skipped_invalid_json == 1

    def test_invalid_records_are_skipped(self, tmp_path):
        # candidate_id="" triggers ValidationError
        lines = [
            json.dumps({"candidate_id": ""}),
            json.dumps(minimal_candidate("CAND_0000002")),
        ]
        path = write_jsonl(lines, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        candidates, result = loader.load_all()
        assert result.parsed == 1
        assert result.skipped_validation == 1

    def test_error_samples_capped(self, tmp_path):
        # Generate more errors than max_error_samples
        bad_lines = [json.dumps({"candidate_id": ""}) for _ in range(20)]
        path = write_jsonl(bad_lines, tmp_path)
        loader = CandidateLoader(path, max_error_samples=5, show_progress=False)
        _, result = loader.load_all()
        assert len(result.validation_errors) == 5

    def test_success_rate_calculation(self, tmp_path):
        lines = [
            json.dumps(minimal_candidate("CAND_0000001")),
            json.dumps(minimal_candidate("CAND_0000002")),
            "{bad json}",
        ]
        path = write_jsonl(lines, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        _, result = loader.load_all()
        assert abs(result.success_rate - 2 / 3) < 1e-9

    def test_stream_is_lazy(self, tmp_path):
        """Streaming should not load all records before iteration starts."""
        data = [minimal_candidate(f"CAND_{i:07d}") for i in range(10)]
        path = write_jsonl(data, tmp_path)
        loader = CandidateLoader(path, show_progress=False)
        it, result = loader.stream()
        # result.parsed should be 0 before iterating
        assert result.parsed == 0
        first = next(it)
        assert first.candidate_id == "CAND_0000000"
        assert result.parsed == 1


class TestLoadResult:
    def test_summary_string(self):
        r = LoadResult(total_lines=100, parsed=95, skipped_blank=2, skipped_invalid_json=1, skipped_validation=2)
        s = r.summary()
        assert "95" in s
        assert "95.00%" in s
