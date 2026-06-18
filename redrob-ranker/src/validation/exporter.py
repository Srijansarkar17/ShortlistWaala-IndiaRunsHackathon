"""
src/validation/exporter.py
===========================
HoneypotExporter
----------------
Writes a list of `DetectionResult` records to a Parquet file.

Schema
------
The dict/list fields (check_reasons, check_penalties, checks_triggered)
are JSON-serialised as strings for maximum Parquet compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .detector import DetectionResult


_COMPRESSION = "snappy"


class HoneypotExporter:
    """
    Writes detection results to a Parquet file.

    Parameters
    ----------
    output_path : str | Path
        Destination file (e.g. ``artifacts/honeypots.parquet``).
    chunk_size : int
        Records per write chunk.
    """

    def __init__(
        self,
        output_path: str | Path,
        chunk_size: int = 10_000,
    ) -> None:
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, results: list[DetectionResult]) -> Path:
        """Bulk export from a list."""
        logger.info(f"Exporting {len(results):,} honeypot records → {self.output_path}")
        writer: pq.ParquetWriter | None = None

        try:
            for chunk in self._chunked(results, self.chunk_size):
                table = self._to_arrow(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.output_path, table.schema, compression=_COMPRESSION
                    )
                writer.write_table(table)
        finally:
            if writer:
                writer.close()

        size_mb = self.output_path.stat().st_size / 1_048_576
        logger.info(
            f"Honeypot Parquet written: {self.output_path} "
            f"({len(results):,} rows, {size_mb:.2f} MB)"
        )
        return self.output_path.resolve()

    def export_stream(
        self,
        results_iter: Iterator[DetectionResult],
    ) -> tuple[Path, int]:
        """Stream export — no full list in memory."""
        writer: pq.ParquetWriter | None = None
        buf: list[DetectionResult] = []
        rows_written = 0

        try:
            for result in results_iter:
                buf.append(result)
                if len(buf) >= self.chunk_size:
                    table = self._to_arrow(buf)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.output_path, table.schema, compression=_COMPRESSION
                        )
                    writer.write_table(table)
                    rows_written += len(buf)
                    buf.clear()

            if buf:
                table = self._to_arrow(buf)
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.output_path, table.schema, compression=_COMPRESSION
                    )
                writer.write_table(table)
                rows_written += len(buf)
        finally:
            if writer:
                writer.close()

        size_mb = self.output_path.stat().st_size / 1_048_576
        logger.info(
            f"Stream export done: {self.output_path} "
            f"({rows_written:,} rows, {size_mb:.2f} MB)"
        )
        return self.output_path.resolve(), rows_written

    @staticmethod
    def read(path: str | Path) -> pd.DataFrame:
        """Read back a honeypots Parquet file as a DataFrame."""
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(result: DetectionResult) -> dict:
        """Convert a DetectionResult to a flat dict for DataFrame creation."""
        return {
            "candidate_id": result.candidate_id,
            "honeypot_score": result.honeypot_score,
            "trust_score": result.trust_score,
            "is_honeypot": result.is_honeypot,
            "total_penalty": result.total_penalty,
            "num_checks_run": result.num_checks_run,
            "num_checks_triggered": result.num_checks_triggered,
            # JSON-serialised complex fields
            "checks_triggered_json": json.dumps(result.checks_triggered),
            "check_reasons_json": json.dumps(result.check_reasons),
            "check_penalties_json": json.dumps(result.check_penalties),
        }

    @staticmethod
    def _to_arrow(records: list[DetectionResult]) -> pa.Table:
        rows = [HoneypotExporter._flatten(r) for r in records]
        df = pd.DataFrame(rows)
        return pa.Table.from_pandas(df, preserve_index=False)

    @staticmethod
    def _chunked(lst: list, size: int) -> Iterator[list]:
        for i in range(0, len(lst), size):
            yield lst[i: i + size]
