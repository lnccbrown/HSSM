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

source_if_present() {
	local file_path="$1"
	if [[ -f "$file_path" ]]; then
		# shellcheck disable=SC1090
		source "$file_path"
	fi
}

bootstrap_environment() {
	local lock_file="$RUN_DIR/submitted/uv-sync.lock"

	source_if_present "$HOME/.bashrc"
	source_if_present "$HOME/.bash_profile"
	source_if_present "$HOME/.profile"

	if command -v module >/dev/null 2>&1 && [[ -n "${HSSM_MODULES:-}" ]]; then
		IFS=',' read -r -a module_list <<< "$HSSM_MODULES"
		for module_name in "${module_list[@]}"; do
			if [[ -n "$module_name" ]]; then
				module load "$module_name"
			fi
		done
	fi

	if ! command -v uv >/dev/null 2>&1; then
		echo "uv is not available in the Slurm job environment." >&2
		echo "Load it in your shell profile or export HSSM_MODULES=<module1,module2> at submission time." >&2
		exit 1
	fi

	mkdir -p "$RUN_DIR/submitted"
	if command -v flock >/dev/null 2>&1; then
		flock "$lock_file" uv sync --group dev --group notebook
	else
		uv sync --group dev --group notebook
	fi
}

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

bootstrap_environment

exec uv run python scripts/notebook_jobs.py run-one \
	--manifest "$MANIFEST_PATH" \
	--run-dir "$RUN_DIR" \
	--index "$NOTEBOOK_INDEX" \
	--fail-on-error
