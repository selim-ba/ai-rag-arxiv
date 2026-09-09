PYTHON ?= python3

.PHONY: install dev test lint fmt ingest index eval clean

install:
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
		|| (echo "Need Python 3.11+. Got: $$($(PYTHON) --version). Try: make install PYTHON=python3.12"; exit 1)
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	@echo ""
	@echo "Done. In VS Code: Cmd+Shift+P -> 'Python: Select Interpreter' -> ./.venv/bin/python"

dev:
	.venv/bin/uvicorn arxiv_rag.api.main:app --reload --port 8000

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests scripts

fmt:
	.venv/bin/ruff format src tests scripts
	.venv/bin/ruff check --fix src tests scripts

ingest:
	.venv/bin/python -m scripts.ingest --ids-file corpus.txt

index:
	.venv/bin/python -m scripts.index

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
