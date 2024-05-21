# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# This Dockerfile is for HSSM
# The buid from the base of scipy-notebook, based on python 3.11

FROM jupyter/scipy-notebook:python-3.11

LABEL maintainer="Hu Chuan-Peng <hcp4715@hotmail.com>"

USER root

RUN apt-get update -y && \
  apt-get upgrade -y && \
  apt-get install -y apt-utils && \
  apt-get install -y build-essential&& \
  apt-get install -y gcc && \
  apt-get install -y g++ && \
  apt-get install -y gfortran && \
  rm -rf /var/lib/apt/lists/*

USER $NB_UID

RUN conda config --add channels  https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
RUN conda install 'python-graphviz' --yes
RUN conda install 'pymc=5.14.0' --yes

RUN pip install -U "jax[cpu]" -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip install hssm -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  fix-permissions "/home/${NB_USER}" && \
  rm -rf ~/.cache/pip

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" &&\
  fix-permissions "/home/${NB_USER}"

# Copy example data and scripts to the example folder
RUN rm -r /home/$NB_USER/work && \
  fix-permissions /home/$NB_USER

COPY /docs/tutorials /home/$NB_USER/tutorials
COPY /tests/fixtures /home/$NB_USER/tutorials/src

USER root
RUN rm -rf /home/$NB_USER/tutorials/main_tutorial

USER $NB_UID
WORKDIR $HOME