#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

RUN_DIR=""
MANIFEST_PATH="notebooks.toml"
PARTITION="batch"
TIME_LIMIT=""
MEMORY=""
CPUS_PER_TASK=""
ACCOUNT=""
JOB_NAME="hssm-notebooks"
DRY_RUN=0
SBATCH_ARGS=()
EXPORT_ITEMS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --cpus-per-task)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --export)
            EXPORT_ITEMS+=("$2")
            shift 2
            ;;
        --sbatch-arg)
            SBATCH_ARGS+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=".cache/notebook-runs/cluster-$(date +%Y%m%dT%H%M%S)"
fi

ARRAY_SPEC=$(uv run python scripts/notebook_jobs.py --manifest "$MANIFEST_PATH" prepare-run --run-dir "$RUN_DIR" --format array)

SBATCH_COMMAND=(
    sbatch
    "--array=$ARRAY_SPEC"
    "--job-name=$JOB_NAME"
    "--output=$RUN_DIR/submitted/slurm-%A_%a.out"
    "--error=$RUN_DIR/submitted/slurm-%A_%a.err"
)

if [[ -n "$PARTITION" ]]; then
    SBATCH_COMMAND+=("--partition=$PARTITION")
fi
if [[ -n "$TIME_LIMIT" ]]; then
    SBATCH_COMMAND+=("--time=$TIME_LIMIT")
fi
if [[ -n "$MEMORY" ]]; then
    SBATCH_COMMAND+=("--mem=$MEMORY")
fi
if [[ -n "$CPUS_PER_TASK" ]]; then
    SBATCH_COMMAND+=("--cpus-per-task=$CPUS_PER_TASK")
fi
if [[ -n "$ACCOUNT" ]]; then
    SBATCH_COMMAND+=("--account=$ACCOUNT")
fi

EXPORT_ITEMS+=("RUN_DIR=$RUN_DIR" "MANIFEST_PATH=$MANIFEST_PATH")
if [[ -n "${PYTENSOR_FLAGS:-}" ]]; then
    EXPORT_ITEMS+=("PYTENSOR_FLAGS=${PYTENSOR_FLAGS}")
fi

if [[ ${#EXPORT_ITEMS[@]} -gt 0 ]]; then
    export_csv="ALL"
    for item in "${EXPORT_ITEMS[@]}"; do
        export_csv+="","$item"
    done
    SBATCH_COMMAND+=("--export=$export_csv")
fi

if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    SBATCH_COMMAND+=("${SBATCH_ARGS[@]}")
fi

SBATCH_COMMAND+=("scripts/submit_notebook_jobs.sh")

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${SBATCH_COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${SBATCH_COMMAND[@]}"
