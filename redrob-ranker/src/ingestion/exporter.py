"""
ParquetExporter
===============
Converts a list of `NormalizedCandidate` records into a Parquet file using
PyArrow for efficient columnar storage.

Design choices
--------------
- Uses dataclasses.asdict for fast dict conversion
- Writes with snappy compression (good balance of speed / size)
- Appends metadata (row count, schema version) to the Parquet file footer
- Supports optional chunked writing for very large candidate pools
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from .normalizer import NormalizedCandidate

_SCHEMA_VERSION = "1.0"
_COMPRESSION = "snappy"


class ParquetExporter:
    """
    Writes normalized candidate records to a Parquet file.

    Parameters
    ----------
    output_path : str | Path
        Destination file (e.g. ``artifacts/candidates.parquet``).
        Parent directories are created automatically.
    chunk_size : int
        Number of records per in-memory chunk during writing.
        Defaults to 10_000; lower this if RAM is constrained.
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

    def export(self, records: list[NormalizedCandidate]) -> Path:
        """
        Write all records to the output Parquet file.

        Parameters
        ----------
        records : list[NormalizedCandidate]

        Returns
        -------
        Path
            Absolute path to the written file.
        """
        logger.info(
            f"Exporting {len(records):,} records → {self.output_path}"
        )
        writer: pq.ParquetWriter | None = None

        try:
            for chunk in self._chunked(records, self.chunk_size):
                table = self._to_arrow_table(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.output_path,
                        table.schema,
                        compression=_COMPRESSION,
                    )
                writer.write_table(table)
                logger.debug(f"  wrote chunk of {len(chunk):,} rows")
        finally:
            if writer is not None:
                writer.close()

        size_mb = self.output_path.stat().st_size / 1_048_576
        logger.info(
            f"Parquet written: {self.output_path} "
            f"({len(records):,} rows, {size_mb:.1f} MB)"
        )
        return self.output_path.resolve()

    def export_stream(
        self,
        records_iter: Iterator[NormalizedCandidate],
        estimated_total: int = 0,
    ) -> tuple[Path, int]:
        """
        Write records from an iterator without buffering all in memory.

        Returns
        -------
        (path, rows_written)
        """
        writer: pq.ParquetWriter | None = None
        buffer: list[NormalizedCandidate] = []
        rows_written = 0

        try:
            for record in records_iter:
                buffer.append(record)
                if len(buffer) >= self.chunk_size:
                    table = self._to_arrow_table(buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            self.output_path,
                            table.schema,
                            compression=_COMPRESSION,
                        )
                    writer.write_table(table)
                    rows_written += len(buffer)
                    logger.debug(f"  streamed chunk, total rows: {rows_written:,}")
                    buffer.clear()

            # flush remaining
            if buffer:
                table = self._to_arrow_table(buffer)
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.output_path,
                        table.schema,
                        compression=_COMPRESSION,
                    )
                writer.write_table(table)
                rows_written += len(buffer)
        finally:
            if writer is not None:
                writer.close()

        size_mb = self.output_path.stat().st_size / 1_048_576
        logger.info(
            f"Stream export done: {self.output_path} "
            f"({rows_written:,} rows, {size_mb:.1f} MB)"
        )
        return self.output_path.resolve(), rows_written

    @staticmethod
    def read(path: str | Path) -> pd.DataFrame:
        """Convenience: read back a candidates Parquet file as a DataFrame."""
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _to_arrow_table(records: list[NormalizedCandidate]) -> pa.Table:
        """Convert a list of NormalizedCandidate to a PyArrow Table."""
        rows = [dataclasses.asdict(r) for r in records]
        df = pd.DataFrame(rows)
        return pa.Table.from_pandas(df, preserve_index=False)

    @staticmethod
    def _chunked(
        lst: list, size: int
    ) -> Iterator[list]:
        """Yield successive fixed-size slices."""
        for i in range(0, len(lst), size):
            yield lst[i : i + size]
