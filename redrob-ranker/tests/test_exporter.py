"""
tests/test_exporter.py
======================
Unit tests for ParquetExporter.

Checks:
- File is created at the specified path
- Row count matches input
- Schema columns match NormalizedCandidate fields
- Stream export works correctly
- Data round-trips (write then read back)
- Output dir is auto-created
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.exporter import ParquetExporter
from src.ingestion.models import Candidate, CandidateProfile, Skill, Certification, CareerEntry
from src.ingestion.normalizer import CandidateNormalizer, NormalizedCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_normalized(cid: str = "CAND_001") -> NormalizedCandidate:
    c = Candidate(
        candidate_id=cid,
        profile=CandidateProfile(
            headline="Engineer",
            summary="5 years experience",
            years_of_experience=5.0,
        ),
        skills=[Skill(name="Python")],
        certifications=[Certification(name="AWS")],
    )
    return CandidateNormalizer().normalize(c)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParquetExporter:
    def test_file_created(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = ParquetExporter(out)
        records = [_make_normalized("CAND_001")]
        exporter.export(records)
        assert out.exists()

    def test_auto_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c" / "out.parquet"
        exporter = ParquetExporter(out)
        exporter.export([_make_normalized()])
        assert out.exists()

    def test_row_count_matches(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = ParquetExporter(out)
        records = [_make_normalized(f"CAND_{i:03d}") for i in range(25)]
        exporter.export(records)
        df = pd.read_parquet(out)
        assert len(df) == 25

    def test_schema_has_expected_columns(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = ParquetExporter(out)
        exporter.export([_make_normalized()])
        df = pd.read_parquet(out)
        expected_cols = {f.name for f in dataclasses.fields(NormalizedCandidate)}
        assert expected_cols.issubset(set(df.columns))

    def test_round_trip_data_integrity(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = ParquetExporter(out)
        record = _make_normalized("CAND_ROUND")
        exporter.export([record])
        df = pd.read_parquet(out)
        row = df[df["candidate_id"] == "CAND_ROUND"].iloc[0]
        assert row["headline"] == "Engineer"
        assert row["num_skills"] == 1
        skills = json.loads(row["skill_names_json"])
        assert "Python" in skills

    def test_chunked_writing(self, tmp_path):
        out = tmp_path / "out.parquet"
        exporter = ParquetExporter(out, chunk_size=10)  # force multi-chunk
        records = [_make_normalized(f"CAND_{i:03d}") for i in range(35)]
        exporter.export(records)
        df = pd.read_parquet(out)
        assert len(df) == 35

    def test_stream_export(self, tmp_path):
        out = tmp_path / "stream.parquet"
        exporter = ParquetExporter(out, chunk_size=5)
        records = [_make_normalized(f"CAND_{i:03d}") for i in range(20)]
        _, rows_written = exporter.export_stream(iter(records))
        assert rows_written == 20
        df = pd.read_parquet(out)
        assert len(df) == 20

    def test_read_static_method(self, tmp_path):
        out = tmp_path / "read_test.parquet"
        exporter = ParquetExporter(out)
        exporter.export([_make_normalized()])
        df = ParquetExporter.read(out)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
