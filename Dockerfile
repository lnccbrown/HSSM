# Runnable HSSM image: HSSM + JupyterLab, ready to explore the tutorials.
#   docker run --rm -p 8888:8888 ghcr.io/lnccbrown/hssm:latest
# Built and published from .github/workflows/docker-image.yml.
#
# Multi-arch (linux/amd64,linux/arm64) is driven by the workflow's `platforms`.
# arm64 works because ssm-simulators (>=0.12.5) and hddm-wfpt (>=0.1.7) now ship
# manylinux aarch64 wheels, so nothing compiles from source on either arch.

# Base bundles pinned uv 0.11.28 + Python 3.13 on a slim Debian (trixie) base,
# so uv and Python come ready — no separate uv copy step needed.
FROM ghcr.io/astral-sh/uv:0.11.28-python3.13-trixie-slim

LABEL org.opencontainers.image.source="https://github.com/lnccbrown/HSSM"
LABEL org.opencontainers.image.description="Bayesian inference for hierarchical sequential sampling models (HSSM) with JupyterLab."
LABEL org.opencontainers.image.licenses="MIT"

# graphviz: the `dot` binary the python `graphviz` package shells out to for model graphs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --uid 1000 hssm
WORKDIR /home/hssm/build

# Install HSSM from this checkout (not PyPI): the image matches the built ref
# exactly and there is no release-timing race. Dependencies still arrive as wheels.
# Deliberately resolves latest-compatible deps (not uv.lock) — this is the runnable
# demo image; the reproducible/locked path is the dev container (`uv sync`).
# ponytail: core + a minimal notebook UI + light zeus-mcmc — enough for 30 of the
# 34 tutorials. The 4 needing the heavy PyTorch/TensorFlow stack (bayesflow/keras,
# sbi, lanfactory) are left out to keep the image slim; see README_DOCKER.md below.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-cache . jupyterlab ipywidgets graphviz zeus-mcmc

# Tutorials to explore, plus a note about the 4 that need extra packages.
COPY docs/tutorials /home/hssm/tutorials
RUN printf '%s\n' \
      '# Running the tutorials in this image' \
      '' \
      '30 of the 34 tutorials run as-is. Four advanced ones need heavy extras that are' \
      'left out to keep the image small:' \
      '' \
      '  - bayesflow_lre_integration, bayesflow_nle_onnx_integration  -> bayesflow, keras' \
      '  - sbi_nre_integration                                        -> sbi, torch' \
      '  - jax_callable_contribution_onnx_example                     -> lanfactory' \
      '' \
      'Install them here:  uv pip install --system bayesflow keras sbi lanfactory' \
      'or use the HSSM dev container (Codespaces / VS Code): uv sync --group notebook' \
      > /home/hssm/tutorials/README_DOCKER.md \
    && chown -R hssm:hssm /home/hssm

USER hssm
WORKDIR /home/hssm/tutorials
EXPOSE 8888

# Secure by default: Jupyter generates a token and prints the full URL (with token)
# to the container logs — copy it from `docker run` output. For a frictionless
# tokenless server on a trusted local network, override the command and append
# `--IdentityProvider.token=`.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
