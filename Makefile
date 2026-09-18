PYTHON ?= python3

# Extra flags for the eval targets, e.g. `make eval ARGS="--agent --limit 5"`.
# `make` swallows anything starting with `--`, so flags have to arrive as a variable.
ARGS ?=

.PHONY: docker-build docker-build-amd64 docker-checks docker-run docker-run-readonly install dev test lint fmt ingest index check-eval eval retrieval-eval grade-eval route-eval followup-eval label-judge score-judge gold-audit gold-apply clean

install:
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" \
		|| (echo "Need Python 3.11+. Got: $$($(PYTHON) --version). Try: make install PYTHON=python3.12"; exit 1)
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev,ingest]"
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

check-eval:
	.venv/bin/python -m scripts.check_eval --ranks

eval:
	.venv/bin/python -m scripts.eval $(ARGS)

retrieval-eval:
	.venv/bin/python -m scripts.retrieval_eval $(ARGS)

grade-eval:
	.venv/bin/python -m scripts.grade_eval $(ARGS)

followup-eval:
	.venv/bin/python -m scripts.followup_eval $(ARGS)

route-eval:
	.venv/bin/python -m scripts.route_eval $(ARGS)

label-judge:
	.venv/bin/python -m scripts.label_judge sample $(ARGS)

score-judge:
	.venv/bin/python -m scripts.label_judge score

gold-audit:
	.venv/bin/python -m scripts.gold_audit propose $(ARGS)

gold-apply:
	.venv/bin/python -m scripts.gold_audit apply

# Stage 6. IMAGE is overridable so a registry path can be passed in Stage 7.
IMAGE ?= arxiv-rag

docker-build:
	docker build -t $(IMAGE) .
	@docker images $(IMAGE) --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"

# Your laptop is arm64; most cloud runtimes are amd64. Built as a separate target rather
# than as the default: cross-building under emulation is slow enough to change how often
# you iterate, so iterate native and cross-build when deploying.
docker-build-amd64:
	docker build --platform linux/amd64 -t $(IMAGE):amd64 .
	@docker images $(IMAGE):amd64 --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"

# The key arrives at runtime, from the environment, never from a layer.
docker-run:
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE)

# How a hardened platform would run it: no writes anywhere, no capabilities, no new
# privileges. If the service needs something it should not, this is where it says so.
docker-run-readonly:
	docker run --rm -p 8000:8000 --env-file .env \
		--read-only --cap-drop ALL --security-opt no-new-privileges $(IMAGE)

# The three failure modes, against the built image, on port 8100 so it does not collide
# with a server you have running.
docker-checks:
	./scripts/container_checks.sh

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
