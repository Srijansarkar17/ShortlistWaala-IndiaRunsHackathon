# Phase 1 Walkthrough — Data Ingestion & Candidate Normalization

## What was built

Phase 1 implements a complete ingestion pipeline for the `redrob-ranker` candidate ranking system. It reads 100,000 raw candidate profiles from `candidates.jsonl`, validates them, normalizes each into a flat searchable document, and writes to a compact Parquet file.

---

## Final Results

| Metric | Value |
|---|---|
| Records read | 100,000 |
| Parsed successfully | **100,000 (100%)** |
| Validation errors | 0 |
| Blank / bad JSON lines | 0 |
| Output Parquet size | **57.6 MB** |
| Total runtime | **4.9 seconds** |

---

## Folder Structure

```
redrob-ranker/
├── rank.py                          ← CLI entry point (Phase 1 pipeline)
├── requirements.txt                 ← pydantic, pandas, pyarrow, loguru, tqdm
│
├── data/
│   └── candidates.jsonl → (symlink) ← 100k candidate profiles
│
├── artifacts/
│   └── candidates.parquet           ← OUTPUT: 100k normalized records, 57.6 MB
│
├── src/
│   ├── __init__.py
│   └── ingestion/
│       ├── __init__.py              ← Exports all public classes
│       ├── models.py                ← Pydantic v2 models
│       ├── loader.py                ← CandidateLoader
│       ├── normalizer.py            ← CandidateNormalizer
│       └── exporter.py              ← ParquetExporter
│
└── tests/
    ├── conftest.py
    ├── test_models.py               ← 24 tests
    ├── test_loader.py               ← 17 tests
    ├── test_normalizer.py           ← 15 tests
    └── test_exporter.py             ← 8 tests   (62 total, all passing)
```

---

## Component Deep-Dive

### 1. `models.py` — Pydantic v2 Dataclass Models

Seven models mapping the real JSON schema:

```
Candidate (root)
 ├── CandidateProfile
 ├── CareerEntry[]
 ├── Education[]
 ├── Skill[]
 ├── Certification[]
 ├── Language[]
 └── RedrobSignals
```

**Key defensive validators:**

| Validator | What it handles |
|---|---|
| `coerce_yoe` | String/int YOE, negative → `None` |
| `parse_nullable_date` | ISO strings, `null`, `""`, malformed dates |
| `strip_strings` | Whitespace-only names become `""` |
| `coerce_duration` | Negative months → `None` |
| `coerce_endorsements` | Negative endorsements clamped to `0` |
| `drop_empty_skills` | Filters skills with `None`/blank names |
| `handle_missing_top_level` | Missing blocks default to `{}` / `[]` |
| `ConfigDict(extra="allow")` | Future `redrob_signals` fields tolerated |

---

### 2. `CandidateLoader` — Streaming JSONL Reader

Memory-efficient — **never loads the full 487 MB file into RAM**.

```python
loader = CandidateLoader("data/candidates.jsonl")
candidates, result = loader.load_all()
print(result.summary())
# Lines read: 100000 | Parsed OK: 100000 | Bad JSON: 0 | Validation errors: 0
```

**Fault tolerance chain:**
```
Line read
  → blank? → skip (skipped_blank++)
  → json.loads fails? → skip (skipped_invalid_json++)
  → Pydantic ValidationError? → skip + sample error (skipped_validation++)
  → OK → yield Candidate
```

**`LoadResult`** tracks all counters and stores up to N validation error samples for debugging.

---

### 3. `CandidateNormalizer` — Text Document Builder

Produces **two outputs per candidate**:

**① Normalized text document** (for embedding / BM25 in Phase 3):
```
[HEADLINE] Backend Engineer | SQL, Spark, Cloud

[SUMMARY] Software / data professional with 6.9 years...

[EXPERIENCE] Implemented streaming data pipelines on Kafka... | Built and maintained...

[SKILLS] Tailwind, NLP, Image Classification, Fine-tuning LLMs, ...

[CERTS] AWS Certified Cloud Practitioner (AWS, 2025)
```
> Sections are **omitted** if the content is empty (no blank `[SKILLS] ` noise).

**② Flat `NormalizedCandidate` record** with:
- Profile scalars (`headline`, `summary`, `years_of_experience`, etc.)
- JSON-encoded list columns (`skill_names_json`, `career_titles_json`, etc.)
- Aggregate counts (`num_skills`, `num_certifications`, etc.)
- Key signals (`notice_period_days`, `open_to_work_flag`, `github_activity_score`, etc.)

---

### 4. `ParquetExporter` — Columnar Storage Writer

Two modes:
- **`export(records)`** — bulk write from a list
- **`export_stream(iterator)`** — streaming write, chunk-by-chunk (used in `rank.py`)

```python
exporter = ParquetExporter("artifacts/candidates.parquet", chunk_size=10_000)
path, rows = exporter.export_stream(normalized_iterator)
```

- **Compression**: Snappy (fast read/write, good compression)
- **Chunking**: 10k rows/chunk prevents OOM on large datasets
- **Auto mkdir**: creates `artifacts/` if missing
- **`ParquetExporter.read(path)`** — convenience method to load back as DataFrame

---

### 5. `rank.py` — CLI Entry Point

Wires the full pipeline: **Loader → Normalizer → Exporter** in a single stream.

```bash
# Default
python rank.py

# Custom paths
python rank.py --input data/candidates.jsonl --output artifacts/candidates.parquet --chunk-size 5000
```

---

## Test Coverage — 62 Tests, All Passing

```
tests/test_models.py    24 tests  ✅
tests/test_loader.py    17 tests  ✅
tests/test_normalizer.py 15 tests ✅
tests/test_exporter.py   8 tests  ✅
```

Notable test scenarios:
- `null` dates → `None` (not crash)
- Negative YOE → `None`
- Empty skill names filtered out
- Blank/bad JSON lines skipped
- Stream laziness (no eager load)
- Parquet round-trip data integrity
- Error sample cap enforced
- Auto-directory creation

---

## Parquet Schema (27 columns)

| Column | Type | Description |
|---|---|---|
| `candidate_id` | string | Primary key |
| `normalized_text` | string | Full searchable document |
| `headline` | string | Profile headline |
| `summary` | string | Profile summary |
| `years_of_experience` | float | Cleaned YOE |
| `current_title` | string | Current job title |
| `skill_names_json` | string | JSON array of skill names |
| `certification_names_json` | string | JSON array of cert names |
| `career_titles_json` | string | JSON array of all titles |
| `career_descriptions_json` | string | JSON array of descriptions |
| `num_skills` | int | Count of skills |
| `num_certifications` | int | Count of certifications |
| `num_career_entries` | int | Count of career roles |
| `notice_period_days` | int | Notice period |
| `open_to_work_flag` | bool | Open to work |
| `github_activity_score` | float | GitHub activity |
| `willing_to_relocate` | bool | Relocation flag |
| ... | ... | + 10 more signal columns |

---

## What Phase 2 can consume

The output `artifacts/candidates.parquet` is ready for:

- **Phase 2 (Honeypot Detection)**: `redrob_signals` scalars + career pattern signals already in the flat record
- **Phase 3 (Feature Engineering)**: All 27 columns + JSON-parsed list fields
- **Phase 4 (Indexing)**: `normalized_text` → BM25 index + dense embedding input
