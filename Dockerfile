# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# HSSM image found at zenkavi/hssm
# docker pull zenkavi/hssm

ARG BASE_CONTAINER=jupyter/minimal-notebook:python-3.9
FROM $BASE_CONTAINER

LABEL maintainer="Zeynep Enkavi <zenkavi@caltech.edu>"

USER root

# ffmpeg for matplotlib anim & dvipng for latex labels
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends ffmpeg dvipng && \
    apt install -y graphviz &&\
    rm -rf /var/lib/apt/lists/*
    
USER $NB_UID

## TODO: Move these to a requirements.txt and install in a virtual env instead of as root
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "scipy==1.10.1" && \
    pip install --no-cache-dir "pymc>=5.6.0" && \
    pip install --no-cache-dir "arviz>=0.14.0" && \
    pip install --no-cache-dir "numpy>=1.23.4" && \
    pip install --no-cache-dir "onnx>=1.12.0" && \
    pip install --no-cache-dir "jax>=0.4.0"  && \
    pip install --no-cache-dir "jaxlib>=0.4.0" && \
    pip install --no-cache-dir "ssm-simulators>=0.3.0" && \
    pip install --no-cache-dir "huggingface-hub>=0.15.1" && \
    pip install --no-cache-dir "onnxruntime>=1.15.0" && \
    pip install --no-cache-dir "bambi>=0.12.0" && \
    pip install --no-cache-dir "pytest>=7.3.1" && \
    pip install --no-cache-dir "black>=23.7.0" && \
    pip install --no-cache-dir "mypy>=1.4.1" && \
    pip install --no-cache-dir "pre-commit>=2.20.0" && \
    pip install --no-cache-dir "jupyterlab>=4.0.2" && \
    pip install --no-cache-dir "ipykernel>=6.16.0" && \
    pip install --no-cache-dir "git+https://github.com/brown-ccv/hddm-wfpt.git" && \
    pip install --no-cache-dir "ipywidgets>=8.0.3" && \
    pip install --no-cache-dir "graphviz>=0.20.1" && \
    pip install --no-cache-dir "ruff>=0.0.272" && \
    pip install --no-cache-dir "numpyro>=0.12.1" && \
    pip install --no-cache-dir "mkdocs>=1.4.3" && \
    pip install --no-cache-dir "mkdocs-material>=9.1.17" && \
    pip install --no-cache-dir "mkdocstrings-python>=1.1.2" && \
    pip install --no-cache-dir "mkdocs-jupyter>=0.24.1" && \
    # pip install --no-cache-dir "hssm>=0.1.2" && \
    pip install git+https://github.com/lnccbrown/HSSM.git && \
    fix-permissions "/home/${NB_USER}"


# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" &&\
     fix-permissions "/home/${NB_USER}"

USER $NB_UID
WORKDIR $HOME
	
# Create a folder for example
RUN mkdir /home/$NB_USER/tutorial_notebooks && \
    mkdir /home/$NB_USER/tutorial_notebooks/no_execute && \
    fix-permissions /home/$NB_USER

# Copy example data and scripts to the example folder
COPY /tutorial_notebooks/tutorial_likelihoods.ipynb /home/${NB_USER}/tutorial_notebooks
COPY /tutorial_notebooks/hugging_face_onnx_models.ipynb /home/${NB_USER}/tutorial_notebooks
COPY /tutorial_notebooks/pymc.ipynb /home/${NB_USER}/tutorial_notebooks
COPY /tutorial_notebooks/no_execute/getting_started.ipynb /home/${NB_USER}/tutorial_notebooks/no_execute
COPY /tutorial_notebooks/no_execute/lapse_prob_and_dist.ipynb /home/${NB_USER}/tutorial_notebooks/no_execute
COPY /tutorial_notebooks/no_execute/main_tutorial.ipynb /home/${NB_USER}/tutorial_notebooks/no_execute
