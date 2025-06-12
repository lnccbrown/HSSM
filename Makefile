run-notebooks:
	@echo "Running notebooks with $$(python --version)"
	@EXIT_CODE=0; \
	for notebook in $$(find docs -name "*.ipynb"); do \
		echo "Cleaning $$notebook"; \
		uv run nb-clean clean -o "$$notebook"; \
		if ! uv run jupyter nbconvert --ExecutePreprocessor.timeout=10000 --to notebook --execute "$$notebook"; then \
			echo "Failed to execute notebook: $$notebook"; \
			EXIT_CODE=1; \
		fi; \
	done; \
	exit $$EXIT_CODE