.PHONY: format lint typecheck test run install clean

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

# Clean cache and build artifacts
clean:
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
