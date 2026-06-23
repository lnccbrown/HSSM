#!/bin/bash
# Simple Slurm job-array runner for notebooks.
# Update the account and array indexes for your cluster before submitting.
#SBATCH --account=default
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --job-name=run-hssm-notebooks
#SBATCH --time=06:00:00
#SBATCH --output=run-hssm-notebooks-%A_%a.out
#SBATCH --error=run-hssm-notebooks-%A_%a.err
#SBATCH --array=0-0

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	REPO_ROOT=$(CDPATH= cd -- "$SLURM_SUBMIT_DIR" && pwd)
else
	SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
	REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
fi

cd "$REPO_ROOT"

RUN_BASE=${SCRATCH:-$HOME}
RUN_DIR=${RUN_DIR:-$RUN_BASE/hssm-notebook-runs/${SLURM_ARRAY_JOB_ID:-manual-run}}
MANIFEST_PATH=${MANIFEST_PATH:-$REPO_ROOT/notebooks.toml}
NOTEBOOK_INDEX=${NOTEBOOK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}

if [[ -z "$NOTEBOOK_INDEX" ]]; then
	echo "SLURM_ARRAY_TASK_ID or NOTEBOOK_INDEX must be set." >&2
	exit 1
fi

if command -v module >/dev/null 2>&1; then
	module load python
	module load gcc
fi

if ! command -v uv >/dev/null 2>&1; then
	echo "uv must already be available on the compute node." >&2
	echo "Install it once for your account or load a module that provides it." >&2
	exit 1
fi

mkdir -p "$RUN_DIR"

if [[ ! -f "$RUN_DIR/inventory.json" ]]; then
	uv run python "$REPO_ROOT/scripts/notebook_jobs.py" \
		--manifest "$MANIFEST_PATH" \
		prepare-run \
		--run-dir "$RUN_DIR" >/dev/null
fi

exec uv run python "$REPO_ROOT/scripts/notebook_jobs.py" \
	--manifest "$MANIFEST_PATH" \
	run-one \
	--run-dir "$RUN_DIR" \
	--index "$NOTEBOOK_INDEX" \
	--fail-on-error
