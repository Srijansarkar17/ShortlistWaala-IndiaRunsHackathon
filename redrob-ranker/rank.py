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
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import CandidateLoader, CandidateNormalizer, ParquetExporter
from src.validation import HoneypotDetector, HoneypotExporter, TrapDetector, TwinResolver
from src.feature_engineering import FeatureStore, JDFeatureGenerator
from src.jd_understanding.parser import JDParser, JDProfile
from src.retrieval.engine import HybridRetriever
from src.ranking import RankingEngine, CrossEncoderReranker
from src.submission import SubmissionGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RedRob Ranker — Phases 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 & 11")
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "all"],
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
    p.add_argument("--jd-features-output", default="artifacts/jd_features.parquet",
                   help="Phase 6 JD Features Parquet output path")
    p.add_argument("--trap-output", default="artifacts/trap_features.parquet",
                   help="Phase 7 Trap Features Parquet output path")
    p.add_argument("--twin-output", default="artifacts/twin_features.parquet",
                   help="Phase 8 Twin Features Parquet output path")
    p.add_argument("--ranking-output", default="artifacts/ranked_candidates.parquet",
                   help="Phase 9 Ranked candidates Parquet output path")
    p.add_argument("--ranking-model-path", default="artifacts/models/lambdarank.txt",
                   help="Phase 9 saved model output path")
    p.add_argument("--rerank-output", default="artifacts/reranked_candidates.parquet",
                   help="Phase 10 Reranked candidates Parquet output path")
    p.add_argument("--rerank-cache-path", default="artifacts/cross_encoder_cache.json",
                   help="Phase 10 Cross-Encoder cache JSON path")
    p.add_argument("--rerank-batch-size", type=int, default=32,
                   help="Phase 10 Cross-Encoder inference batch size")
    p.add_argument("--submission-output", default="submission.csv",
                   help="Phase 11 final Submission CSV output path")
    p.add_argument("--train-only", action="store_true",
                   help="Only run ranking model training, do not output ranking predictions")
    p.add_argument("--rank-only", action="store_true",
                   help="Only run ranking model inference using saved model, do not retrain")
    p.add_argument("--filter-honeypots", action="store_true",
                   help="Filter out flagged honeypots in Phase 6")
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
# Phase 6
# ---------------------------------------------------------------------------

def run_phase6(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 6: JD Feature Generator")
    logger.info("=" * 60)

    # 1. Load JD Profile
    logger.info(f"[1/5] Loading JD Profile from: {args.jd_output}")
    jd_profile_path = Path(args.jd_output)
    if not jd_profile_path.exists():
        logger.error(f"JD Profile JSON not found at: {jd_profile_path}. Please run Phase 4 first.")
        sys.exit(1)
        
    with open(jd_profile_path, "r", encoding="utf-8") as fh:
        jd_profile = JDProfile.model_validate_json(fh.read())

    # 2. Check Candidate parquet
    logger.info(f"[2/5] Checking Candidates database: {args.candidates_parquet}")
    candidates_path = Path(args.candidates_parquet)
    if not candidates_path.exists():
        logger.error(f"Candidates database Parquet not found at: {candidates_path}. Please run Phase 1 first.")
        sys.exit(1)

    # 3. Check precomputed Features
    logger.info(f"[3/5] Checking Candidate Features: {args.features_output}")
    features_path = Path(args.features_output)
    if not features_path.exists():
        logger.error(f"Candidate Features Parquet not found at: {features_path}. Please run Phase 3 first.")
        sys.exit(1)

    # 4. Check Honeypots
    logger.info(f"[4/5] Checking Honeypots database: {args.honeypot_output}")
    honeypot_path = Path(args.honeypot_output)
    if not honeypot_path.exists():
        logger.error(f"Honeypots database Parquet not found at: {honeypot_path}. Please run Phase 2 first.")
        sys.exit(1)

    # 5. Check Retrieval Results
    logger.info(f"[5/5] Checking Retrieval Results: {args.retrieval_output}")
    retrieval_path = Path(args.retrieval_output)
    if not retrieval_path.exists():
        logger.error(f"Retrieval results Parquet not found at: {retrieval_path}. Please run Phase 5 first.")
        sys.exit(1)

    logger.info("Generating JD features matrix...")
    generator = JDFeatureGenerator()
    features_df = generator.generate(
        jd_profile=jd_profile,
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        honeypots_parquet=honeypot_path,
        retrieval_parquet=retrieval_path,
        filter_honeypots=args.filter_honeypots
    )

    out_path = Path(args.jd_features_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"JD features generated successfully")
    logger.info(f"Output: {out_path} | Candidates featured: {len(features_df):,} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 7
# ---------------------------------------------------------------------------

def run_phase7(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 7: Trap Detection")
    logger.info("=" * 60)

    # 1. Check Candidate parquet
    logger.info(f"[1/3] Checking Candidates database: {args.candidates_parquet}")
    candidates_path = Path(args.candidates_parquet)
    if not candidates_path.exists():
        logger.error(f"Candidates database Parquet not found at: {candidates_path}. Please run Phase 1 first.")
        sys.exit(1)

    # 2. Check precomputed Features
    logger.info(f"[2/3] Checking Candidate Features: {args.features_output}")
    features_path = Path(args.features_output)
    if not features_path.exists():
        logger.error(f"Candidate Features Parquet not found at: {features_path}. Please run Phase 3 first.")
        sys.exit(1)

    logger.info("[3/3] Running trap detection engine...")
    detector = TrapDetector()
    
    # Use current datetime as reference date
    ref_date = datetime(2026, 6, 21)
    
    trap_df = detector.detect(
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        reference_date=ref_date
    )

    out_path = Path(args.trap_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trap_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Trap features generated successfully")
    logger.info(f"Output: {out_path} | Candidates processed: {len(trap_df):,} | Time: {elapsed:.1f}s")


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

    if args.phase in ("6", "all"):
        run_phase6(args)

    if args.phase in ("7", "all"):
        run_phase7(args)

    if args.phase in ("8", "all"):
        run_phase8(args)

    if args.phase in ("9", "all"):
        run_phase9(args)

    if args.phase in ("10", "all"):
        run_phase10(args)

    if args.phase in ("11", "all"):
        run_phase11(args)


# ---------------------------------------------------------------------------
# Phase 8
# ---------------------------------------------------------------------------

def run_phase8(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 8: Twin Resolver")
    logger.info("=" * 60)

    # 1. Check Candidate parquet
    logger.info(f"[1/3] Checking Candidates database: {args.candidates_parquet}")
    candidates_path = Path(args.candidates_parquet)
    if not candidates_path.exists():
        logger.error(f"Candidates database Parquet not found at: {candidates_path}. Please run Phase 1 first.")
        sys.exit(1)

    # 2. Check precomputed Features
    logger.info(f"[2/3] Checking Candidate Features: {args.features_output}")
    features_path = Path(args.features_output)
    if not features_path.exists():
        logger.error(f"Candidate Features Parquet not found at: {features_path}. Please run Phase 3 first.")
        sys.exit(1)

    logger.info("[3/3] Running twin resolver engine...")
    resolver = TwinResolver()
    
    # Use reference date
    ref_date = datetime(2026, 6, 21)
    
    twin_df = resolver.resolve(
        candidates_parquet=candidates_path,
        features_parquet=features_path,
        reference_date=ref_date
    )

    out_path = Path(args.twin_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    twin_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Twin resolution finished successfully")
    logger.info(f"Output: {out_path} | Candidates processed: {len(twin_df):,} | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 9
# ---------------------------------------------------------------------------

def run_phase9(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 9: Ranking Engine")
    logger.info("=" * 60)

    # Check input file requirements
    for path_str, label in [
        (args.candidates_parquet, "Candidates"),
        (args.features_output, "Features"),
        (args.honeypot_output, "Honeypots"),
        (args.retrieval_output, "Retrieval"),
        (args.jd_features_output, "JD Features"),
        (args.trap_output, "Trap Features"),
        (args.twin_output, "Twin Features"),
    ]:
        p = Path(path_str)
        if not p.exists():
            logger.error(f"{label} database Parquet not found at: {p}. Please run the respective phase first.")
            sys.exit(1)

    engine = RankingEngine()
    
    # 1. Compile dataset
    df_compiled = engine.compile_dataset(
        candidates_parquet=args.candidates_parquet,
        features_parquet=args.features_output,
        honeypots_parquet=args.honeypot_output,
        retrieval_parquet=args.retrieval_output,
        jd_features_parquet=args.jd_features_output,
        trap_features_parquet=args.trap_output,
        twin_features_parquet=args.twin_output,
    )

    # 2. Train if requested/needed
    should_train = not args.rank_only
    if should_train:
        engine.train(df_compiled, model_output_path=args.ranking_model_path)
    
    # 3. Predict if requested
    should_rank = not args.train_only
    if should_rank:
        ranked_df = engine.rank(df_compiled, model_path=args.ranking_model_path)
        
        # Select top 500
        top_500 = ranked_df.head(500)
        
        out_path = Path(args.ranking_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        top_500.to_parquet(out_path, index=False)
        
        elapsed = time.perf_counter() - t0
        logger.info("-" * 60)
        logger.info(f"Ranking finished successfully")
        logger.info(f"Output: {out_path} | Top 500 candidates ranked | Time: {elapsed:.1f}s")
    else:
        elapsed = time.perf_counter() - t0
        logger.info("-" * 60)
        logger.info(f"Training finished successfully | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 10
# ---------------------------------------------------------------------------

def run_phase10(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 10: Cross Encoder Reranker")
    logger.info("=" * 60)

    # Check input files requirements
    for path_str, label in [
        (args.ranking_output, "Ranking engine output (top 500)"),
        (args.candidates_parquet, "Candidates database"),
        (args.jd_input, "Job Description text file"),
    ]:
        p = Path(path_str)
        if not p.exists():
            logger.error(f"{label} not found at: {p}. Please run respective phase first.")
            sys.exit(1)

    reranker = CrossEncoderReranker(batch_size=args.rerank_batch_size)
    
    reranked_df = reranker.rerank(
        top_n_parquet=args.ranking_output,
        candidates_parquet=args.candidates_parquet,
        jd_txt_path=args.jd_input,
        cache_json_path=args.rerank_cache_path,
        top_k=500
    )

    out_path = Path(args.rerank_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reranked_df.to_parquet(out_path, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Cross-Encoder reranking finished successfully")
    logger.info(f"Output: {out_path} | Top 100 candidates ranked | Time: {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Phase 11
# ---------------------------------------------------------------------------

def run_phase11(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Phase 11: Submission Generator")
    logger.info("=" * 60)

    # Check input files requirements
    for path_str, label in [
        (args.rerank_output, "Cross-Encoder reranked output (top 100)"),
        (args.candidates_parquet, "Candidates database"),
        (args.features_output, "Features"),
        (args.honeypot_output, "Honeypots"),
        (args.retrieval_output, "Retrieval"),
        (args.jd_features_output, "JD Features"),
        (args.trap_output, "Trap Features"),
        (args.twin_output, "Twin Features"),
    ]:
        p = Path(path_str)
        if not p.exists():
            logger.error(f"{label} not found at: {p}. Please run respective phase first.")
            sys.exit(1)

    # Set up generator
    generator = SubmissionGenerator()
    
    # We output to args.submission_output (e.g. submission.csv)
    out_path = Path(args.submission_output)

    final_csv_path = generator.generate(
        reranked_parquet=args.rerank_output,
        candidates_parquet=args.candidates_parquet,
        features_parquet=args.features_output,
        honeypots_parquet=args.honeypot_output,
        retrieval_parquet=args.retrieval_output,
        jd_features_parquet=args.jd_features_output,
        trap_features_parquet=args.trap_output,
        twin_features_parquet=args.twin_output,
        output_csv_path=out_path
    )

    elapsed = time.perf_counter() - t0
    logger.info("-" * 60)
    logger.info(f"Submission generation and validation finished successfully!")
    logger.info(f"Final output: {final_csv_path.resolve()} | Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()




