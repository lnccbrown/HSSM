# Runnable HSSM image: HSSM + JupyterLab, ready to explore the tutorials.
#   docker run --rm -p 8888:8888 ghcr.io/lnccbrown/hssm:latest
# Built and published from .github/workflows/docker-image.yml.
#
# Multi-arch (linux/amd64,linux/arm64) is driven by the workflow's `platforms`.
# arm64 works because ssm-simulators (>=0.12.5) and hddm-wfpt (>=0.1.7) now ship
# manylinux aarch64 wheels, so nothing compiles from source on either arch.

FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/lnccbrown/HSSM"
LABEL org.opencontainers.image.description="Bayesian inference for hierarchical sequential sampling models (HSSM) with JupyterLab."
LABEL org.opencontainers.image.licenses="MIT"

# graphviz: the `dot` binary the python `graphviz` package shells out to for model graphs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

# uv for fast installs (ecosystem standard). Pinned for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

# Non-root user.
RUN useradd --create-home --uid 1000 hssm
WORKDIR /home/hssm/build

# Install HSSM from this checkout (not PyPI): the image matches the built ref
# exactly and there is no release-timing race. Dependencies still arrive as wheels.
# Deliberately resolves latest-compatible deps (not uv.lock) — this is the runnable
# demo image; the reproducible/locked path is the dev container (`uv sync`).
# ponytail: core + a minimal notebook UI only. The heavy optional stacks
# (bayesflow, sbi, lanfactory, keras) are NOT included — add them at run time
# (`uv pip install ...`) or in a follow-up "full" image if a tutorial needs them.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-cache . jupyterlab ipywidgets graphviz

# Tutorials to explore. Some advanced notebooks need the heavy stacks above.
COPY docs/tutorials /home/hssm/tutorials
RUN chown -R hssm:hssm /home/hssm

USER hssm
WORKDIR /home/hssm/tutorials
EXPOSE 8888

# Secure by default: Jupyter generates a token and prints the full URL (with token)
# to the container logs — copy it from `docker run` output. For a frictionless
# tokenless server on a trusted local network, override the command and append
# `--IdentityProvider.token=`.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
