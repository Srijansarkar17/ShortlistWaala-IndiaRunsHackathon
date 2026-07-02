# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile.sandbox — Runs the actual ranking code with Stage 3 constraints
# ─────────────────────────────────────────────────────────────────────────────
# This builds the full ranking environment and runs rank.py to verify:
#   ✔ Runtime ≤ 5 minutes wall-clock
#   ✔ Memory ≤ 16 GB RAM (enforced by --memory=16g at docker run)
#   ✔ CPU only — no GPU (default Docker, no --gpus flag)
#   ✔ Network OFF (enforced by --network none at docker run)
#   ✔ Disk ≤ 5 GB intermediate state
#
# USAGE:
#   docker build -t redrob-sandbox -f Dockerfile.sandbox .
#
#   docker run --rm \
#     --network none \
#     --memory="16g" \
#     -v "$(pwd):/workspace" \
#     redrob-sandbox \
#     --candidates /workspace/data/candidates.jsonl \
#     --out /workspace/submission_test.csv
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL description="Redrob Hackathon — Full ranking sandbox (Stage 3 simulation)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code, models, and artifacts
COPY . /app/

# Default entrypoint: run the ranking pipeline
ENTRYPOINT ["python", "rank.py"]
