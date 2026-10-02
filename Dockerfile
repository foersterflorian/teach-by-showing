# teach-by-showing container image. Rationale: DECISIONS.md §10.
#
#   docker compose build
#   docker compose up            # then open http://localhost:8081 on the host
#
# The image runs exactly the versions pdm.lock pins - the ones the test suite
# checked - on a digest-pinned Python base. It needs no GPU, no display and no
# devices: the browser on the host draws the crane and reads the gamepad.

# Same digest in both stages, and the same as feature-viz's, so that the two
# images on the stand machine share their base layers.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

# pdm only turns the lockfile into a hashed requirements list; it does not
# reach the runtime image. Pinned so that the export itself is reproducible.
RUN pip install --no-cache-dir pdm==2.29.2

WORKDIR /build
COPY pyproject.toml pdm.lock ./
RUN pdm export --prod -o requirements.txt \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.txt

# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="teach-by-showing" \
      org.opencontainers.image.source="https://github.com/foersterflorian/teach-by-showing" \
      org.opencontainers.image.licenses="MIT"

COPY --from=build /opt/venv /opt/venv
COPY src /opt/teach-by-showing/src
COPY LICENSE /opt/teach-by-showing/LICENSE

# Unprivileged. /data is where runs, agents and the gallery will live; the
# volume mounted there inherits this ownership on first use.
RUN useradd --system --uid 10001 --no-create-home teach-by-showing \
 && install -d -o teach-by-showing /data
USER teach-by-showing
WORKDIR /opt/teach-by-showing

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/teach-by-showing/src \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

EXPOSE 8081

# /healthz is 503 when the simulation loop has stopped ticking (§8).
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=2).status != 200)"]

# 0.0.0.0 inside the container; compose publishes the port to the host's
# localhost only.
ENTRYPOINT ["python", "-m", "teach_by_showing.api", "--host", "0.0.0.0"]
