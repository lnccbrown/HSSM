# Installation

**Important Update:** From HSSM 0.2.1 on, we recommend `conda` as the virtual environment manager for HSSM. We will also gradually make HSSM available directly through `conda-forge` in the near future. For now, please follow the instruction below to install HSSM:

## Recommended: install HSSM in a conda environment

### Step 1: Create a conda environment

If you haven't already, please follow the [Anaconda official website](https://www.anaconda.com/download) to install conda. We assume that you already have one of [Anaconda](https://www.anaconda.com/download), [Miniconda](https://docs.anaconda.com/free/miniconda/index.html), [miniforge](https://github.com/conda-forge/miniforge/releases), or [mambaforge](https://github.com/conda-forge/miniforge/releases) installed on your system and have access to either `conda` or `mamba` available on your command line.

To create a conda environment, use the following command. Substitute `mamba` for `conda` if `mamba` is available:

```bash
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
```

Substitute `<your-env-name>` with the name of the virtual environment that you choose. HSSM 0.2.0 and above supports Python versions 3.10 and 3.11.

### Step 2: Install PyMC through conda-forge

Installation through `conda-forge` is the official way of installing PyMC. This will also install other libraries such as `libblas` that PyMC requires to run properly.

```bash
conda install -c conda-forge pymc
```

### [Optional] Install JAX with CUDA support

If you need to sample with GPU, please install JAX with GPU support after installing PyMC following one of the two commands below:

#### Option 1: Install JAX with CUDA support via `conda`

At the moment, there is a community supported conda build on conda-forge:

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

#### Option 2: Install JAX with CUDA support via `pip`

Installing `jax` via `pip` should also work:

```bash
pip install jax[cuda12]
```

#### Support for Apple Silicon, AMD, and other GPUs

JAX also has support other GPUs. Please follow the [Official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to install the correct version of JAX before proceeding.

### Step 3: Install `hssm` through `pip`

In the same environment, install `hssm` through `pip`.

```bash
pip install hssm
```

## Advanced Installation

### Install HSSM directly with Pip

`hssm` is also available through PyPI. You can directly install it with pip into any virtual environment via:

```bash
pip install hssm
```

!!! note

    While this installation is much simpler, you might need optional dependencies to use JAX-based samplers and to produce model graphs. You might also encounter this warning message `WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.` Please refer to our [advanced installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/) for more details.

### Install the dev version of HSSM

You can also install the bleeding-edge version of `hssm` directly from this repo:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

### Install HSSM on Google Colab

The good news is that Google Colab comes with PyMC and JAX pre-configured. That holds true even if you are using the GPU and TPU backend, so you simply need to install HSSM via pip on Colab regardless of the backend you are using:

```bash
!pip install hssm
```

## Install optional dependencies

Whether you have installed HSSM via `conda`, `pip`, or GitHub, you might still need additional
packages installed for additional features such as sampling with `blackjax` or GPU support
for `JAX`. Please follow the instructions below if you need any of these additional
features:

### 1. Sampling with JAX through `numpyro` or `blackjax`

JAX-based sampling is done through `numpyro` and `blackjax`. `numpyro` is installed as
a dependency by default. You need to have `blackjax` installed if you want to use the
`nuts_blackjax` sampler.

```bash
pip install blackjax
```

### 2. Visualizing the model with `graphviz`

Model graphs are created with `model.graph()` through `graphviz`. If you have installed
hssm in a conda environment, you can simply install `graphviz` in conda:

```bash
conda install -c conda-forge graphviz
```

If you have installed hssm in a non-conda environment, you need to have `graphviz` installed system-wide and then install its Python binding:

#### Install graphviz system-wide

Please follow the instructions on the
[graphviz official site](https://graphviz.org/download/) to install graphviz for your
specific platform.

#### Install graphviz python binding

Once graphviz is installed, you can install its Python binding via pip:

```bash
pip install graphviz
```

## Common issues

1. I run into warnings such as

```
WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
```

   This is because `pytensor`, the compute backend of PyMC, cannot find a BLAS library on your
   system to optimize its computation. This can be resolved by following the recommended
   steps to install HSSM into a conda environment. If conda cannot be used, you can follow
   [this discussion](https://github.com/pymc-devs/pytensor/issues/524) to link a BLAS library
   with `pytensor`.

2. `pip` installation fails with missing dependencies:

   Here's an example:

   ```
   ERROR: Could not find a version that satisfies the requirement jaxlib<0.5.0,>=0.4.0 (from hssm) (from versions: none)
   ERROR: No matching distribution found for jaxlib<0.5.0,>=0.4.0 (from hssm)
   ```

   HSSM has very specific requirements for the versions of `jax`, `pymc`, and `bambi`.
   This problem can usually be resolved by installing HSSM into a dedicated virtual
   environment.

!!! note

    Possible solutions to any issues with installations with hssm can be located [here](https://github.com/lnccbrown/HSSM/discussions). Also feel free to start a new
    discussion thread if you don't find answers there. We recommend installing HSSM into
    a new conda environment with Python 3.10 or 3.11 to prevent any problems with dependencies
    during the installation process. Please note that hssm is only tested for python 3.10,
    3.11. As of HSSM v0.2.0, support for Python 3.9 is dropped. Use unsupported python
    versions with caution.

## Questions?

If you have any questions, please
[open an issue](https://github.com/lnccbrown/HSSM/issues) in our
[GitHub repo](https://github.com/lnccbrown/HSSM).
