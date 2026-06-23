MANIFEST ?= notebooks.toml
RUN_DIR ?= .cache/notebook-runs/cluster-$(shell date +%Y%m%dT%H%M%S)
PARTITION ?= cpu
TIME_LIMIT ?= 08:00:00
MEM ?= 16G
CPUS_PER_TASK ?=
ACCOUNT ?=
JOB_NAME ?= hssm-notebooks
NOTEBOOK ?=
NOTEBOOK_INDEX ?=
SBATCH_ARGS ?=
SBATCH_EXPORTS ?=

.PHONY: notebook-discover notebook-prepare notebook-submit-dry-run notebook-submit notebook-aggregate notebook-rerun

notebook-discover:
	uv run python scripts/notebook_jobs.py --manifest $(MANIFEST) discover --include-disabled

notebook-prepare:
	uv run python scripts/notebook_jobs.py --manifest $(MANIFEST) prepare-run --run-dir $(RUN_DIR)

notebook-submit-dry-run:
	bash scripts/sbatch_submit_notebook_jobs.sh \
		--manifest $(MANIFEST) \
		--run-dir $(RUN_DIR) \
		--partition $(PARTITION) \
		--time-limit $(TIME_LIMIT) \
		--mem $(MEM) \
		$(if $(CPUS_PER_TASK),--cpus-per-task $(CPUS_PER_TASK)) \
		$(if $(ACCOUNT),--account $(ACCOUNT)) \
		$(if $(JOB_NAME),--job-name $(JOB_NAME)) \
		$(foreach arg,$(SBATCH_ARGS),--sbatch-arg $(arg)) \
		$(foreach arg,$(SBATCH_EXPORTS),--export $(arg)) \
		--dry-run

notebook-submit:
	bash scripts/sbatch_submit_notebook_jobs.sh \
		--manifest $(MANIFEST) \
		--run-dir $(RUN_DIR) \
		--partition $(PARTITION) \
		--time-limit $(TIME_LIMIT) \
		--mem $(MEM) \
		$(if $(CPUS_PER_TASK),--cpus-per-task $(CPUS_PER_TASK)) \
		$(if $(ACCOUNT),--account $(ACCOUNT)) \
		$(if $(JOB_NAME),--job-name $(JOB_NAME)) \
		$(foreach arg,$(SBATCH_ARGS),--sbatch-arg $(arg)) \
		$(foreach arg,$(SBATCH_EXPORTS),--export $(arg))

notebook-aggregate:
	uv run python scripts/notebook_jobs.py aggregate --run-dir $(RUN_DIR)

notebook-rerun:
	@if [ -n "$(NOTEBOOK)" ]; then \
		uv run python scripts/notebook_jobs.py --manifest $(MANIFEST) run-one --run-dir $(RUN_DIR) --notebook $(NOTEBOOK) --fail-on-error; \
	elif [ -n "$(NOTEBOOK_INDEX)" ]; then \
		uv run python scripts/notebook_jobs.py --manifest $(MANIFEST) run-one --run-dir $(RUN_DIR) --index $(NOTEBOOK_INDEX) --fail-on-error; \
	else \
		echo "Set NOTEBOOK=<path> or NOTEBOOK_INDEX=<n>" >&2; \
		exit 2; \
	fi
