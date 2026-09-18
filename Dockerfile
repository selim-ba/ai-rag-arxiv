# Step 2 of Stage 6: the simplest image that runs. One stage, no optimisation.
# Steps 3-5 split the build, order the layers for caching, and drop root - each of those
# is a separate commit with a measurement attached, because "I made it smaller" without a
# number before and after is how this project has been wrong before.
#
# python:3.12-slim, not 3.14: every dependency here (numpy, pymupdf, tiktoken) ships a
# prebuilt 3.12 wheel. On 3.14 any one of them without a wheel compiles from source and
# every rebuild costs twenty minutes. The project declares requires-python >= 3.11, so
# 3.12 is inside its own contract; the container and the laptop run different minors, and
# that is a thing to know rather than a thing to fix.
FROM python:3.12-slim

# PYTHONUNBUFFERED is not cosmetic here. Python buffers stdout when it is not a terminal,
# which is exactly the case inside a container - the JSON request lines Stage 5 exists to
# produce would arrive in 8 KB bursts, or not at all if the process died first.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, in a layer keyed only on pyproject.toml, then the source. Ordered
# this way because Docker invalidates every layer after the first changed one: with the
# source copied first, editing a single line of Python re-ran the whole dependency
# install - measured at 15.5s, paid on every rebuild for the rest of this stage.
#
# The dependency list is read out of pyproject.toml at build time rather than duplicated
# into a requirements.txt. A second list is a second thing to forget: the image would keep
# building happily while installing something the project no longer declares. `tomllib` is
# standard library from 3.11, so this costs nothing.
#
# The `[ingest]` extra is deliberately NOT installed. pymupdf (63 MB) and tiktoken (4 MB)
# are imported only by the ingestion path, which a container that answers questions never
# runs. `tests/test_packaging.py` is what keeps that true.
#
# `.venv` is excluded by .dockerignore: it holds macOS arm64 wheels, useless here.
COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); \
print('\n'.join(d['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

# --no-deps: the dependencies are already in place from the layer above, and without it
# pip would resolve the whole tree again on every source change.
COPY src/ ./src/
RUN pip install --no-deps .

# The index, and only the index: 874 chunks and a 5 MB matrix. Baked in rather than
# mounted, so the image version IS the data version - a container that starts is a
# container with the corpus it was tested against. `data/papers/` (272 MB of PDFs) stays
# out: the index is the artefact, the PDFs are how it was made.
COPY data/index/ ./data/index/

# `config.py` resolves data_dir relative to the working directory, so this is where
# `data/index` must land, and where the embedding cache will want to write. That write is
# deliberately not solved yet - step 5 runs the container read-only to find it.

# 0.0.0.0, not 127.0.0.1: a server bound to loopback inside a container is unreachable
# from outside it, and the symptom is an empty reply rather than an error.
EXPOSE 8000
CMD ["uvicorn", "arxiv_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
