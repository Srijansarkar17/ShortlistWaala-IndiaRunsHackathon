"""
tests/test_honeypot_exporter.py
================================
Unit tests for HoneypotExporter.

Tests:
- File created with correct row count
- Schema has expected columns
- JSON fields are valid JSON
- Stream export works
- Round-trip data integrity
- Auto directory creation
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.models import Candidate, CandidateProfile, RedrobSignals, Skill
from src.validation.detector import HoneypotDetector
from src.validation.exporter import HoneypotExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(cid: str = "CAND_001"):
    c = Candidate(
        candidate_id=cid,
        profile=CandidateProfile(years_of_experience=5.0),
    )
    return HoneypotDetector().detect(c)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHoneypotExporter:
    def test_file_created(self, tmp_path):
        out = tmp_path / "honeypots.parquet"
        exporter = HoneypotExporter(out)
        exporter.export([_make_result()])
        assert out.exists()

    def test_auto_creates_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "honeypots.parquet"
        exporter = HoneypotExporter(out)
        exporter.export([_make_result()])
        assert out.exists()

    def test_row_count(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = HoneypotExporter(out)
        results = [_make_result(f"CAND_{i:03d}") for i in range(20)]
        exporter.export(results)
        df = pd.read_parquet(out)
        assert len(df) == 20

    def test_schema_columns_present(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = HoneypotExporter(out)
        exporter.export([_make_result()])
        df = pd.read_parquet(out)
        expected = {
            "candidate_id", "honeypot_score", "trust_score", "is_honeypot",
            "total_penalty", "num_checks_run", "num_checks_triggered",
            "checks_triggered_json", "check_reasons_json", "check_penalties_json",
        }
        assert expected.issubset(set(df.columns))

    def test_json_fields_parseable(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = HoneypotExporter(out)
        exporter.export([_make_result()])
        df = pd.read_parquet(out)
        row = df.iloc[0]
        assert isinstance(json.loads(row["checks_triggered_json"]), list)
        assert isinstance(json.loads(row["check_reasons_json"]), dict)
        assert isinstance(json.loads(row["check_penalties_json"]), dict)

    def test_round_trip_scores(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = HoneypotExporter(out)
        result = _make_result("CAND_ROUNDTRIP")
        exporter.export([result])
        df = pd.read_parquet(out)
        row = df.iloc[0]
        assert row["candidate_id"] == "CAND_ROUNDTRIP"
        assert abs(row["honeypot_score"] - result.honeypot_score) < 1e-6
        assert abs(row["trust_score"] - result.trust_score) < 1e-6

    def test_stream_export_row_count(self, tmp_path):
        out = tmp_path / "stream.parquet"
        exporter = HoneypotExporter(out, chunk_size=5)
        results = [_make_result(f"CAND_{i:03d}") for i in range(23)]
        _, rows = exporter.export_stream(iter(results))
        assert rows == 23
        df = pd.read_parquet(out)
        assert len(df) == 23

    def test_read_static_method(self, tmp_path):
        out = tmp_path / "read.parquet"
        exporter = HoneypotExporter(out)
        exporter.export([_make_result()])
        df = HoneypotExporter.read(out)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
