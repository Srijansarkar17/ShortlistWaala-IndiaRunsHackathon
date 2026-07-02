# ShortlistWaala — Candidate Discovery & Ranking Challenge

This repository contains the candidate discovery, ranking, and scoring pipeline built by team **ShortlistWaala** for the Redrob hackathon.

---

## Methodology Overview

We implement a robust multi-stage candidate discoverer and ranker:

1. **Phase 1 & 2: Ingestion & Honeypot Detection**: Normalized candidates data, identified anomalous fields (impossible date intervals, salary inversions, zero-duration expert skills) to create a clean database and flag honeypot records.
2. **Phase 3 & 4: Feature Engineering & JD Parsing**: Generated candidate behavioural signals (recruiter response rates, average tenure, connections, GitHub activity). Parsed the job description rules, required skills, and boundaries.
3. **Phase 5: Hybrid Retrieval**: Blended lexical (BM25) and semantic (Dense Embeddings using `all-MiniLM-L6-v2`) retrieval using Reciprocal Rank Fusion (RRF) to pull the top 5,000 candidates.
4. **Phase 6, 7 & 8: JD Matching, Traps, & Twins**: Scanned the candidate pool for traps (consulting-only, research-only, LangChain wrapper focus, job hoppers, keyword stuffers) and resolved identical twin profiles.
5. **Phase 9: LambdaRank model**: Trained a LightGBM LambdaRank model across 5 distinct recruiter personas to capture general suitability.
6. **Phase 10: Cross-Encoder Reranking**: Used the `cross-encoder/ms-marco-MiniLM-L-6-v2` transformer model to score semantic relevance against the JD on the top 500 candidates.
7. **Phase 11: Multi-Factor Adjusted Final Score**: Refined the raw Cross-Encoder scores with strict penalties:
   * **Experience Penalty**: Strict linear decay ($0.5$ per year away from the boundaries) if YoE is outside the requested 5–9 years range.
   * **Technical Role Check**: Flagged and heavily penalized ($-50$ score deduction) candidates whose headlines/titles represent non-technical fields (e.g. Civil, Mechanical, HR, Graphic Design, Customer Support, Accountant) or lack technical software engineering/ML designations.
   * **Trap & Notice Period Penalties**: Down-weighted candidates based on trap confidence scores and notice periods exceeding 30 days.
   * **Behavioral Boost**: Boosted candidates based on Redrob engagement signals.

---

## Setup & Installation

Ensure you are using Python 3.9+ in your environment.

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenMP warning (macOS)**:
   The pipeline executes CPU-optimized LightGBM. On macOS, make sure OpenMP is installed if you encounter library loading warnings:
   ```bash
   brew install libomp
   ```

---

## How to Reproduce Submission

### Local Execution
To run the entire pipeline end-to-end and generate the final validated `submission.csv` at the repository root:

```bash
python3 rank.py --phase all --candidates ./data/candidates.jsonl --out ./submission.csv
```

Individual phases can also be executed:
* `python3 rank.py --phase 9` (rank and train LightGBM)
* `python3 rank.py --phase 10` (cross-encoder rerank)
* `python3 rank.py --phase 11` (generate and validate submission)

### Docker Execution (Organizers Sandbox)
To reproduce the submission inside the sandboxed Docker container (satisfying $\le 5$ min runtime, $\le 16$ GB RAM, CPU-only, and offline constraints):

1. **Build the Sandbox Image**:
   ```bash
   docker build -t redrob-sandbox .
   ```

2. **Run the Sandbox Container**:
   ```bash
   docker run --rm \
     --network none \
     --memory="16g" \
     -v "$(pwd):/workspace" \
     -w /workspace \
     redrob-sandbox \
     --candidates /workspace/data/candidates.jsonl \
     --out /workspace/submission.csv
   ```

---

## Validation

To run the comprehensive submission and repository layout validation checks:

```bash
python3 validate_submission.py --csv submission.csv --candidates data/candidates.jsonl --repo-root .
```
