"""
rank.py — Phases 1 & 2 entry point
=====================================
Phase 1: Load → Normalize → Parquet (candidates.parquet)
Phase 2: Load normalized → Detect honeypots → Parquet (honeypots.parquet)

Usage
-----
    python rank.py [--phase 1|2|3|all]
                   [--input PATH]
                   [--output PATH]
                   [--honeypot-output PATH]
                   [--features-output PATH]
                   [--chunk-size N]
                   [--honeypot-threshold FLOAT]

Defaults:
  --phase             all
  --input             data/candidates.jsonl
  --output            artifacts/candidates.parquet
  --honeypot-output   artifacts/honeypots.parquet
  --features-output   artifacts/features.parquet
  --chunk-size        10000
  --honeypot-threshold 0.6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import CandidateLoader, CandidateNormalizer, ParquetExporter
from src.validation import HoneypotDetector, HoneypotExporter
from src.feature_engineering.store import FeatureStore
from src.jd_understanding.parser import JDParser, JDProfile
from src.retrieval.engine import HybridRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RedRob Ranker — Phases 1, 2, 3, 4, & 5")
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "5", "all"],
                   help="Which phase(s) to run (default: all)")
    p.add_argument("--input", default="data/candidates.jsonl",
                   help="Path to candidates.jsonl")
    p.add_argument("--output", default="artifacts/candidates.parquet",
                   help="Phase 1 Parquet output path")
    p.add_argument("--honeypot-output", default="artifacts/honeypots.parquet",
                   help="Phase 2 Parquet output path")
    p.add_argument("--features-output", default="artifacts/features.parquet",
                   help="Phase 3 Parquet output path")
    p.add_argument("--jd-input", default="job_description.txt",
                   help="Phase 4 Job Description text path")
    p.add_argument("--jd-output", default="artifacts/jd_profile.json",
                   help="Phase 4 JSON output path")
    p.add_argument("--candidates-parquet", default="artifacts/candidates.parquet",
                   help="Candidates Parquet database path")
    p.add_argument("--retrieval-output", default="artifacts/retrieval_results.parquet",
                   help="Phase 5 Retrieval results Parquet output path")
    p.add_argument("--embeddings-cache-dir", default="artifacts/embeddings_cache",
                   help="Directory for caching candidate dense embeddings")
    p.add_argument("--chunk-size", type=int, default=10_000,
                   help="Rows per write chunk")
    p.add_argument("--honeypot-threshold", type=float, default=0.6,
                   help="Score above which a candidate is flagged as honeypot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def run_phase1(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 1: Data Ingestion & Normalization")
    logger.info("=" * 60)

    logger.info(f"[1/3] Loading: {args.input}")
    loader = CandidateLoader(path=args.input, show_progress=True)
    candidate_iter, load_result = loader.stream()

    logger.info("[2/3] Normalizing...")
    normalizer = CandidateNormalizer()

    def normalized_stream():
        for c in candidate_iter:
            yield normalizer.normalize(c)

    logger.info(f"[3/3] Exporting → {args.output}")
    exporter = ParquetExporter(output_path=args.output, chunk_size=args.chunk_size)
    _, rows_written = exporter.export_stream(normalized_stream())

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(load_result.summary())
    logger.info(f"Rows written: {rows_written:,} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def run_phase2(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 2: Honeypot Detection")
    logger.info("=" * 60)

    logger.info(f"[1/3] Loading candidates from: {args.input}")
    loader = CandidateLoader(path=args.input, show_progress=True)
    candidate_iter, load_result = loader.stream()

    logger.info(f"[2/3] Running honeypot detection (threshold={args.honeypot_threshold})...")
    detector = HoneypotDetector(threshold=args.honeypot_threshold)
    exporter = HoneypotExporter(
        output_path=args.honeypot_output, chunk_size=args.chunk_size
    )

    honeypot_count = 0

    def detection_stream():
        nonlocal honeypot_count
        for candidate in candidate_iter:
            result = detector.detect(candidate)
            if result.is_honeypot:
                honeypot_count += 1
            yield result

    logger.info(f"[3/3] Exporting → {args.honeypot_output}")
    _, rows_written = exporter.export_stream(detection_stream())

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(load_result.summary())
    logger.info(f"Honeypots detected: {honeypot_count:,} / {rows_written:,} "
                f"({honeypot_count/max(rows_written,1)*100:.2f}%)")
    logger.info(f"Output: {args.honeypot_output} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

def run_phase3(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 3: Feature Engineering")
    logger.info("=" * 60)

    logger.info(f"[1/3] Loading candidates from: {args.input}")
    loader = CandidateLoader(path=args.input, show_progress=True)
    candidate_iter, load_result = loader.stream()

    logger.info(f"[2/3] Extracting features...")
    store = FeatureStore(output_path=args.features_output, chunk_size=args.chunk_size)

    logger.info(f"[3/3] Exporting → {args.features_output}")
    _, rows_written = store.export_stream(candidate_iter)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(load_result.summary())
    logger.info(f"Features extracted for {rows_written:,} candidates")
    logger.info(f"Output: {args.features_output} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 4
# ---------------------------------------------------------------------------

def run_phase4(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 4: Job Description Understanding")
    logger.info("=" * 60)

    logger.info(f"[1/3] Reading job description from: {args.jd_input}")
    jd_path = Path(args.jd_input)
    if not jd_path.exists():
        logger.error(f"Job Description file not found at: {jd_path}")
        sys.exit(1)

    with open(jd_path, "r", encoding="utf-8") as fh:
        jd_text = fh.read()

    logger.info("[2/3] Extracting metadata locally...")
    parser = JDParser()
    profile = parser.parse(jd_text)

    logger.info(f"[3/3] Exporting to: {args.jd_output}")
    out_path = Path(args.jd_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        # Pydantic v2 format dump
        fh.write(profile.model_dump_json(indent=2))

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Job Description parsed successfully")
    logger.info(f"Output: {out_path} | Time: {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------

def run_phase5(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 5: Hybrid Retrieval")
    logger.info("=" * 60)

    logger.info(f"[1/4] Loading JD Profile from: {args.jd_output}")
    jd_profile_path = Path(args.jd_output)
    if not jd_profile_path.exists():
        logger.error(f"JD Profile JSON not found at: {jd_profile_path}. Please run Phase 4 first.")
        sys.exit(1)
        
    with open(jd_profile_path, "r", encoding="utf-8") as fh:
        jd_profile = JDProfile.model_validate_json(fh.read())

    logger.info(f"[2/4] Loading candidates database from: {args.candidates_parquet}")
    candidates_path = Path(args.candidates_parquet)
    if not candidates_path.exists():
        logger.error(f"Candidates database Parquet not found at: {candidates_path}. Please run Phase 1 first.")
        sys.exit(1)
        
    candidates_df = pd.read_parquet(candidates_path)
    logger.info(f"Loaded {len(candidates_df):,} candidate profiles")

    logger.info("[3/4] Running retrieval pipeline...")
    retriever = HybridRetriever()
    results_df = retriever.retrieve(
        jd_profile=jd_profile,
        candidates_df=candidates_df,
        top_n=5000,
        embeddings_cache_dir=args.embeddings_cache_dir
    )

    logger.info(f"[4/4] Saving top 5,000 results to: {args.retrieval_output}")
    out_path = Path(args.retrieval_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Hybrid retrieval finished successfully")
    logger.info(f"Output: {out_path} | Candidates retrieved: {len(results_df):,} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.phase in ("1", "all"):
        run_phase1(args)

    if args.phase in ("2", "all"):
        run_phase2(args)

    if args.phase in ("3", "all"):
        run_phase3(args)

    if args.phase in ("4", "all"):
        run_phase4(args)

    if args.phase in ("5", "all"):
        run_phase5(args)


if __name__ == "__main__":
    main()
