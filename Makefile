PYTHON ?= .venv/bin/python

.PHONY: setup check lint format test submission

setup:
	python3.12 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

check: lint test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m unittest linker/test_link.py research/test_frontier_charts.py -v

# Regenerate submission.csv (an empty new-sessions file = the original unlabeled sessions only).
submission:
	$(PYTHON) -c "import pandas as pd; pd.read_parquet('sessions.parquet').iloc[:0].to_parquet('/tmp/empty_sessions.parquet')"
	$(PYTHON) linker/link.py /tmp/empty_sessions.parquet submission.csv
