# `make all` runs on the real panel (params.data.path in config.yaml) and fails with
# E_DATA_FILE_MISSING if the file is not there. `make all-sample` is the quick path on
# synthetic data; its output is never a source of results.
SHELL := /bin/bash
export PYTHONPATH := src
PYTHON ?= python

.PHONY: install install-editable lint test sample check-sample all-sample all clean

install:
	$(PYTHON) -m pip install -r requirements.txt

# Optional: puts `fcst` on the import path through the package metadata, so PYTHONPATH is not
# needed outside make. Editable on purpose: config.yaml and templates/ are read from the
# project root, not from site-packages.
install-editable:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests

sample:
	$(PYTHON) -m fcst.data --write-sample

check-sample:
	$(PYTHON) -m fcst.data --check-sample

all-sample: lint test sample check-sample
	$(PYTHON) -m fcst.pipeline --sample

all: lint test
	$(PYTHON) -m fcst.pipeline

clean:
	find artifacts reports -mindepth 1 -delete 2>/dev/null || true
	rm -f data/sample.csv
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
