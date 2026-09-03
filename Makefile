# Development tasks for jMetalPy.

.PHONY: help test test-verbose test-coverage lint format package clean-build install-dev docs docs-build

help:
	@echo "Available commands:"
	@echo "  test         run all tests"
	@echo "  test-verbose run tests with verbose output"
	@echo "  test-coverage run tests with coverage report"
	@echo "  lint         run ruff linting (blocking)"
	@echo "  format       format code with ruff"
	@echo "  package      build sdist and wheel, then check them with twine"
	@echo "  clean-build  clean build artifacts"
	@echo "  install-dev  install development dependencies"
	@echo "  docs         serve the documentation locally with live reload"
	@echo "  docs-build   build the documentation, failing on any warning"

test:
	python -m pytest tests/ -x

test-verbose:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest --cov=src/jmetal --cov-report=html --cov-report=term tests/

lint:
	python -m ruff check src/ tests/ examples/

format:
	python -m ruff format src/ tests/ examples/
	python -m ruff check --fix src/ tests/ examples/

package:
	python -m build
	python -m twine check dist/*

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

install-dev:
	pip install -e ".[dev]"

docs:
	mkdocs serve

docs-build:
	mkdocs build --strict
