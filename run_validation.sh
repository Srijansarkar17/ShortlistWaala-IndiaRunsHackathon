#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_validation.sh — Build & run the Redrob submission validator
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./run_validation.sh                          # Quick CSV-only validation
#   ./run_validation.sh --full                   # Full validation with candidates.jsonl
#   ./run_validation.sh --sandbox                # Full Stage 3 sandbox simulation
#   ./run_validation.sh --run-rank               # Run rank.py then validate
#   ./run_validation.sh --help                   # Show help
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Colours ──
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Defaults ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_PATH="${SCRIPT_DIR}/submission.csv"
CANDIDATES_PATH=""
IMAGE_NAME="redrob-validator"
MODE="quick"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv)
            CSV_PATH="$2"
            shift 2
            ;;
        --candidates)
            CANDIDATES_PATH="$2"
            shift 2
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --sandbox)
            MODE="sandbox"
            shift
            ;;
        --run-rank)
            MODE="run-rank"
            shift
            ;;
        --help|-h)
            echo ""
            echo "Redrob Hackathon v4 — Submission Validation Script"
            echo ""
            echo "Usage: ./run_validation.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --csv PATH          Path to submission CSV (default: ./submission.csv)"
            echo "  --candidates PATH   Path to candidates.jsonl"
            echo "  --full              Full validation (CSV + candidates + repo structure)"
            echo "  --sandbox           Full Stage 3 sandbox simulation (--network none, --memory 16g)"
            echo "  --run-rank          Run rank.py inside sandbox, then validate output"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_validation.sh"
            echo "  ./run_validation.sh --csv my_submission.csv --full"
            echo "  ./run_validation.sh --sandbox --candidates /path/to/candidates.jsonl"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ── Banner ──
echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  REDROB HACKATHON v4 — SUBMISSION VALIDATION${NC}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Mode:        ${BOLD}${MODE}${NC}"
echo -e "  CSV:         ${CSV_PATH}"
echo -e "  Candidates:  ${CANDIDATES_PATH:-'(not provided)'}"
echo -e "  Repo Root:   ${SCRIPT_DIR}"
echo ""

# ── Step 1: Build Docker image ──
echo -e "${BOLD}[1/3] Building Docker image '${IMAGE_NAME}'...${NC}"
docker build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile.validate" "${SCRIPT_DIR}"
echo -e "${GREEN}✔ Image built successfully${NC}"
echo ""

# ── Step 2: Prepare Docker run command ──
echo -e "${BOLD}[2/3] Preparing sandbox environment...${NC}"

DOCKER_ARGS=(
    "docker" "run" "--rm"
)

# Mount workspace
DOCKER_ARGS+=("-v" "${SCRIPT_DIR}:/workspace")

# Mount candidates if provided
if [[ -n "${CANDIDATES_PATH}" && -f "${CANDIDATES_PATH}" ]]; then
    DOCKER_ARGS+=("-v" "${CANDIDATES_PATH}:/workspace/candidates.jsonl:ro")
    echo -e "  Mounting candidates: ${CANDIDATES_PATH}"
fi

# Mode-specific Docker flags
case ${MODE} in
    sandbox|run-rank)
        DOCKER_ARGS+=("--network" "none")
        DOCKER_ARGS+=("--memory=16g")
        echo -e "  ${YELLOW}Network: BLOCKED (--network none)${NC}"
        echo -e "  ${YELLOW}Memory:  16 GB limit (--memory=16g)${NC}"
        ;;
    *)
        echo -e "  Network: Allowed (use --sandbox for full simulation)"
        echo -e "  Memory:  Host default (use --sandbox for 16 GB limit)"
        ;;
esac

DOCKER_ARGS+=("${IMAGE_NAME}")

# Entrypoint args
CSV_FILENAME=$(basename "${CSV_PATH}")
DOCKER_ARGS+=("--csv" "/workspace/${CSV_FILENAME}")

if [[ -n "${CANDIDATES_PATH}" && -f "${CANDIDATES_PATH}" ]]; then
    DOCKER_ARGS+=("--candidates" "/workspace/candidates.jsonl")
fi

DOCKER_ARGS+=("--repo-root" "/workspace")

# Run-rank mode
if [[ "${MODE}" == "run-rank" ]]; then
    DOCKER_ARGS+=("--run-rank")
    DOCKER_ARGS+=("--rank-cmd" "python /workspace/rank.py --candidates /workspace/candidates.jsonl --out /workspace/${CSV_FILENAME}")
fi

echo ""

# ── Step 3: Run validation ──
echo -e "${BOLD}[3/3] Running validation...${NC}"
echo -e "  Command: ${DOCKER_ARGS[*]}"
echo ""

START_TIME=$(date +%s)

"${DOCKER_ARGS[@]}"
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${NC}"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}  ✔ ALL VALIDATION CHECKS PASSED (${DURATION}s)${NC}"
else
    echo -e "${RED}${BOLD}  ✘ VALIDATION FAILED — Fix issues above before submitting (${DURATION}s)${NC}"
fi
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════════${NC}"
echo ""

exit ${EXIT_CODE}
