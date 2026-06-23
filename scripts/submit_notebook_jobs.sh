#!/usr/bin/env bash
# Slurm batch payload for one notebook array task.
#SBATCH --job-name=hssm-notebooks
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
# These default log paths are overridden by scripts/sbatch_submit_notebook_jobs.sh.
#SBATCH --output=.cache/notebook-runs/slurm-%A_%a.out
#SBATCH --error=.cache/notebook-runs/slurm-%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

cd "$REPO_ROOT"

RUN_DIR=${RUN_DIR:-}
MANIFEST_PATH=${MANIFEST_PATH:-notebooks.toml}
NOTEBOOK_INDEX=${NOTEBOOK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}

if [[ -z "$RUN_DIR" ]]; then
	echo "RUN_DIR must be set before submitting this script." >&2
	echo "Prepare a run first with: uv run python scripts/notebook_jobs.py prepare-run --run-dir <path> --format array" >&2
	exit 1
fi

if [[ -z "$NOTEBOOK_INDEX" ]]; then
	echo "SLURM_ARRAY_TASK_ID or NOTEBOOK_INDEX must be set." >&2
	exit 1
fi

exec uv run python scripts/notebook_jobs.py run-one \
	--manifest "$MANIFEST_PATH" \
	--run-dir "$RUN_DIR" \
	--index "$NOTEBOOK_INDEX" \
	--fail-on-error
