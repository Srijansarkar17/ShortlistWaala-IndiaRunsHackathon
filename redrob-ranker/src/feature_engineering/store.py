import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_engineering.extractor import FeatureExtractor
from src.ingestion.models import Candidate

class FeatureStore:
    """Orchestrates feature extraction and Parquet export."""
    
    def __init__(self, output_path: str, chunk_size: int = 10_000):
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.extractor = FeatureExtractor()
        
    def export_stream(self, candidate_stream: Iterator[Candidate]) -> tuple[Path, int]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunk = []
        rows_written = 0
        writer = None
        
        try:
            for candidate in candidate_stream:
                features = self.extractor.extract(candidate)
                chunk.append(features)
                
                if len(chunk) >= self.chunk_size:
                    writer = self._write_chunk(chunk, writer)
                    rows_written += len(chunk)
                    chunk.clear()
                    
            if chunk:
                writer = self._write_chunk(chunk, writer)
                rows_written += len(chunk)
                chunk.clear()
                
        finally:
            if writer:
                writer.close()
                
        return self.output_path, rows_written
        
    def _write_chunk(self, chunk: list[dict], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        df = pd.DataFrame(chunk)
        table = pa.Table.from_pandas(df)
        
        if writer is None:
            writer = pq.ParquetWriter(self.output_path, table.schema, compression='snappy')
            
        writer.write_table(table)
        return writer
