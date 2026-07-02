# Runnable HSSM image: HSSM + JupyterLab, ready to explore the tutorials.
#   docker run --rm -p 8888:8888 ghcr.io/lnccbrown/hssm:latest
# Built and published from .github/workflows/docker-image.yml.
#
# ponytail: amd64 only for now. ssm-simulators and hddm-wfpt ship no linux/arm64
# wheels yet, so a native arm64 build would have to compile them from source
# (GSL/OpenMP). Apple Silicon runs this image fine under emulation. Flip to
# multi-arch by adding linux/arm64 to the workflow's `platforms` once both
# packages publish aarch64 Linux wheels (tracked upstream).

FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/lnccbrown/HSSM"
LABEL org.opencontainers.image.description="Bayesian inference for hierarchical sequential sampling models (HSSM) with JupyterLab."
LABEL org.opencontainers.image.licenses="MIT"

# graphviz: the `dot` binary the python `graphviz` package shells out to for model graphs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs (ecosystem standard).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root user.
RUN useradd --create-home --uid 1000 hssm
WORKDIR /home/hssm/build

# Install HSSM from this checkout (not PyPI): the image matches the built ref
# exactly and there is no release-timing race. Dependencies still arrive as wheels.
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

# ponytail: tokenless is convenient for a local demo container; the server only
# listens inside the container's network. Set a token if you expose the port publicly.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token="]
