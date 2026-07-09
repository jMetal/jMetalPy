# Minimal makefile for Sphinx documentation and development tasks
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = jMetalPy
SOURCEDIR     = docs/source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional development commands:"
	@echo "  test         run all tests"
	@echo "  test-verbose run tests with verbose output"
	@echo "  test-coverage run tests with coverage report"
	@echo "  lint         run ruff linting (blocking)"
	@echo "  format       format code with ruff"
	@echo "  typecheck    run mypy (informative, not a merge gate)"
	@echo "  package      build sdist and wheel, then check them with twine"
	@echo "  clean-build  clean build artifacts"
	@echo "  install-dev  install development dependencies"

.PHONY: help Makefile test test-verbose test-coverage lint format typecheck package clean-build install-dev

# Development commands
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

typecheck:
	python -m mypy src/jmetal --ignore-missing-imports

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

# "make github" option to build gh-pages
github:
	@make html
	@cp -a $(BUILDDIR)/html/. docs
	@rm -r $(BUILDDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp -R $(BUILDDIR)/html/* docs

