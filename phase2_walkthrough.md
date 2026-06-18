# Phase 2 Walkthrough — Honeypot Detection Engine

## What was built

Phase 2 implements a fully deterministic rule-based engine that scans every candidate profile for fabricated or impossible data signals. Detected candidates receive a `honeypot_score` and `trust_score` stored in `artifacts/honeypots.parquet`.

---

## Live Results (100k candidates)

| Metric | Value |
|---|---|
| Candidates scanned | 100,000 |
| Honeypots detected | **623 (0.62%)** |
| Output Parquet size | **1.77 MB** |
| Total runtime | **4.1 seconds** |
| Tests | **63 / 63 passing** |

---

## Folder Structure (Phase 2 additions)

```
redrob-ranker/
├── rank.py                                ← Updated: --phase 1|2|all
├── src/
│   └── validation/
│       ├── __init__.py
│       ├── checks.py                      ← [NEW] 7 deterministic check functions
│       ├── detector.py                    ← [NEW] HoneypotDetector orchestrator
│       └── exporter.py                    ← [NEW] HoneypotExporter → Parquet
├── artifacts/
│   └── honeypots.parquet                  ← OUTPUT: 100k records, 1.77 MB
└── tests/
    ├── test_checks.py                     ← [NEW] 41 tests for all 7 checks
    ├── test_detector.py                   ← [NEW] 14 tests for detector
    └── test_honeypot_exporter.py          ← [NEW] 8 tests for exporter
```

---

## The 7 Deterministic Checks

| ID | Check | Trigger Condition | Penalty |
|---|---|---|---|
| **C1** | Salary Inversion | `salary_min > salary_max` | 0.30 |
| **C2** | Job Date Inversion | Any non-current role where `start_date > end_date` | 0.25 / violation (cap 0.50) |
| **C3** | Education Date Inversion | Any edu entry where `start_year > end_year` | 0.15 / violation (cap 0.30) |
| **C4** | Duration Mismatch | `|declared_months - computed_months| > 3` | 0.10 / violation (cap 0.40) |
| **C5** | Expert Zero Duration | ≥ 3 "advanced"/"expert" skills with `duration_months == 0` | 0.20 (flat batch) |
| **C6** | Skill Exceeds Experience | Any skill `duration > (YOE × 12) + 12` | 0.20 / violation (cap 0.40) |
| **C7** | Startup Founding Violation | Claimed employment at a startup before it was founded | 0.35 / violation (cap 0.70) |

### Startup Registry (C7)

| Company | Founded |
|---|---|
| OpenAI | 2015 |
| HuggingFace | 2016 |
| Pinecone | 2019 |
| Cohere | 2019 |
| Anthropic | 2021 |
| Qdrant | 2021 |
| Perplexity | 2022 |
| LangChain | 2022 |
| LlamaIndex | 2022 |
| Mistral | 2023 |

---

## Scoring Model

```
raw_penalty      = Σ penalty_i  for each triggered check
honeypot_score   = clip(raw_penalty, 0.0, 1.0)
trust_score      = 1.0 - honeypot_score
is_honeypot      = honeypot_score > threshold  (default: 0.6)
```

**Design properties:**
- Scores are always in `[0, 1]` — monotone, calibrated
- `honeypot_score + trust_score == 1.0` always
- Additive penalties: multiple violations compound
- Per-check caps prevent a single runaway check from dominating

---

## Component Deep-Dive

### `checks.py` — 7 Pure Check Functions

Each check is a standalone pure function: `check_*(candidate: Candidate) → CheckResult`.

```python
@dataclass
class CheckResult:
    check: CheckName      # enum identifier
    triggered: bool       # did it fire?
    penalty: float        # how much to deduct
    reason: str           # human-readable explanation
    details: list[str]    # per-violation detail strings
```

All penalty weights and thresholds are **constants at the top of the file** for easy tuning:

```python
PENALTY_SALARY_INVERSION    = 0.30
PENALTY_JOB_DATE_INVERSION  = 0.25
DURATION_MISMATCH_TOLERANCE = 3      # months
EXPERT_ZERO_DUR_THRESHOLD   = 3      # min count to trigger C5
```

### `detector.py` — `HoneypotDetector`

Runs all 7 checks and produces a `DetectionResult`:

```python
detector = HoneypotDetector(threshold=0.6)
result = detector.detect(candidate)
print(result.honeypot_score, result.trust_score, result.is_honeypot)
```

```python
@dataclass
class DetectionResult:
    candidate_id: str
    honeypot_score: float        # [0, 1]
    trust_score: float           # 1 - honeypot_score
    is_honeypot: bool
    checks_triggered: list[str]  # which checks fired
    total_penalty: float         # raw pre-clip sum
    num_checks_run: int          # always 7
    num_checks_triggered: int
    check_reasons: dict[str, str]    # all 7 reasons
    check_penalties: dict[str, float] # only triggered ones
```

### `exporter.py` — `HoneypotExporter`

Stream-writes `DetectionResult` objects to Parquet in 10k-row chunks.

```python
exporter = HoneypotExporter("artifacts/honeypots.parquet")
path, rows = exporter.export_stream(detection_iter)
```

---

## Parquet Schema (10 columns)

| Column | Type | Description |
|---|---|---|
| `candidate_id` | string | Primary key |
| `honeypot_score` | float64 | [0, 1] — higher = more suspicious |
| `trust_score` | float64 | 1 - honeypot_score |
| `is_honeypot` | bool | True if score > threshold |
| `total_penalty` | float64 | Raw pre-clip penalty sum |
| `num_checks_run` | int | Always 7 |
| `num_checks_triggered` | int | Number of checks that fired |
| `checks_triggered_json` | string | JSON array of check names |
| `check_reasons_json` | string | JSON dict of all 7 reasons |
| `check_penalties_json` | string | JSON dict of triggered penalties |

---

## Running Phase 2

```bash
# Phase 2 only (uses data/candidates.jsonl → artifacts/honeypots.parquet)
python rank.py --phase 2

# Both phases
python rank.py --phase all

# Custom threshold
python rank.py --phase 2 --honeypot-threshold 0.5
```

---

## Test Coverage — 63 Tests, All Passing

### `test_checks.py` — 41 tests

| Check | Tests |
|---|---|
| C1 Salary Inversion | 7 |
| C2 Job Date Inversion | 6 |
| C3 Education Date Inversion | 4 |
| C4 Duration Mismatch | 6 |
| C5 Expert Zero Duration | 5 |
| C6 Skill Exceeds Experience | 5 |
| C7 Startup Founding Year | 8 |

**Key scenarios covered:**
- Clean → never triggers
- All boundary conditions (equal min/max, exact founding year, within tolerance)
- Penalty caps enforced
- None/missing fields always safe-default to no trigger
- `current_job` flag respected in C2

### `test_detector.py` — 14 tests

- `honeypot_score == 0, trust_score == 1` for clean candidate
- `score + trust == 1.0` always (mathematical invariant)
- Score clipped to `[0, 1]`
- `threshold=1.0` → nothing flagged; `threshold=0.0` → everything flagged
- `num_checks_run == 7` always
- `check_penalties` only contains keys for triggered checks
- `detect_batch` produces correct count

### `test_honeypot_exporter.py` — 8 tests

- File creation + auto-mkdir
- Row count matches input
- All 10 schema columns present
- JSON fields parseable
- Round-trip data integrity (scores match)
- Stream chunked export
- `HoneypotExporter.read()` utility

---

## What Phase 3 can consume

`artifacts/honeypots.parquet` is JOIN-able to `candidates.parquet` on `candidate_id`:

- **Feature Engineering**: `honeypot_score` and `trust_score` become ranking features
- **Ranking (Phase 4)**: Candidates with `is_honeypot=True` are penalised to the bottom
- **Post-processing**: Final submission enforces honeypot penalty on score outputs
