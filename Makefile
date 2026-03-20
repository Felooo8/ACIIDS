PYTHON ?= python

install:
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src/aciids tests examples

test:
	pytest

check: lint test
