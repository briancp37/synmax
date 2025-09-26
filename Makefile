.PHONY: format lint typecheck test run install clean agent-plan agent-run agent-ask

# Install dependencies
install:
	poetry install

# Format code
format:
	poetry run black src/ tests/ --line-length 100
	poetry run ruff check src/ tests/ --fix

# Lint code
lint:
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/ --line-length 100

# Type check
typecheck:
	poetry run mypy src/

# Run tests
test:
	poetry run pytest tests/ -v --cov=src/data_agent --cov-report=term-missing

# Run the CLI
run:
	poetry run python -m data_agent.cli

# Agent shortcuts
agent-plan:
	@echo "Usage: make agent-plan QUERY='your question here' [MODEL=model_name] [EXPORT=filename.json]"
	@if [ -z "$(QUERY)" ]; then \
		echo "Error: QUERY parameter is required"; \
		echo "Example: make agent-plan QUERY='detect anomalies in Texas pipelines' MODEL=claude-sonnet EXPORT=my_plan.json"; \
		exit 1; \
	fi
	poetry run python -m data_agent.cli plan "$(QUERY)" $(if $(MODEL),--model $(MODEL)) $(if $(EXPORT),--export $(EXPORT)) --dry-run

agent-run:
	@echo "Usage: make agent-run PLAN=plan_file.json [EXPORT=results.json] [MATERIALIZE=heavy]"
	@if [ -z "$(PLAN)" ]; then \
		echo "Error: PLAN parameter is required"; \
		echo "Example: make agent-run PLAN=my_plan.json EXPORT=results.json MATERIALIZE=heavy"; \
		exit 1; \
	fi
	poetry run python -m data_agent.cli run --plan $(PLAN) $(if $(EXPORT),--export $(EXPORT)) $(if $(MATERIALIZE),--materialize $(MATERIALIZE))

agent-ask:
	@echo "Usage: make agent-ask QUERY='your question here' [MODEL=model_name] [EXPORT=filename.json]"
	@if [ -z "$(QUERY)" ]; then \
		echo "Error: QUERY parameter is required"; \
		echo "Example: make agent-ask QUERY='top 10 pipelines by volume' MODEL=gpt-4.1 EXPORT=auto"; \
		exit 1; \
	fi
	poetry run python -m data_agent.cli ask "$(QUERY)" $(if $(MODEL),--model $(MODEL)) $(if $(EXPORT),--export $(EXPORT))

# Clean cache and build artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
