# Installation

**Update**: From HSSM 0.2.1 on, we recommend `conda` as the virtual environment manager for HSSM. We will also gradually make HSSM available directly through `conda-forge` in the near future. For now, please follow the instruction below to install HSSM:

## Recommended: install HSSM in a conda environment

### Step 1: Create a conda environment

If you haven't already, please follow the [Anaconda official website](https://www.anaconda.com/download) to install anaconda. We assume that you already have one of [Anaconda](https://www.anaconda.com/download), [Miniconda](https://docs.anaconda.com/free/miniconda/index.html), [miniforge](https://github.com/conda-forge/miniforge/releases), or [mambaforge](https://github.com/conda-forge/miniforge/releases) installed on your system and have access to either `conda` or `mamba` available on your command line.

To create a conda environment, use the following command. Substitute `mamba` for `conda` if `mamba` is available:

```bash
conda create -n <your-env-name> python=3.11
conda activate <your-env-name>
```

Substitute `<your-env-name>` with the name of the virtual environment that you choose. HSSM 0.2.0 and above support Python versions 3.10 and 3.11.

### Step 2: Install PyMC through conda-forge

Installation through `conda-forge` is the official way of installing PyMC. This will also install other libraries such as `libblas` that PyMC requires to run properly.

```bash
conda install -c conda-forge pymc
```

As of HSSM 0.2.1, HSSM supports PyMC 5.10.4. If a future newer version of PyMC causes compatibility issues, please specify the version of PyMC:

```bash
conda install -c conda-forge pymc=5.10
```

### Step 3: Install `hssm` through `pip`

In the same environment, install `hssm` through `pip`.

```bash
pip install hssm
```

## Advanced: install via `pip` from PyPI or GitHub

`hssm` is also available through PyPI. You can directly install it with pip into any virtual environment via:

```bash
pip install hssm
```

You can also install the bleeding-edge version of `hssm` directly from this repo:

```
pip install git+https://github.com/lnccbrown/HSSM.git
```

Because HSSM depends on very specific versions of PyMC, JAX and Bambi, we recommend that
you install HSSM into a dedicated virtual environment to avoid dependency conflicts.

## Advanced: install optional dependencies

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

### 2. Sampling with JAX support for GPU

The `nuts_numpyro` sampler uses JAX as the backend and thus can support sampling on nvidia
GPUs. The only thing you need to do to take advantage of this is to install JAX with CUDA
support before installing HSSM. This works whether `conda` is used or not. Here's one example:

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install hssm
```

The example above shows how to install JAX with CUDA 11 support. Please refer to the
[JAX Installation](https://jax.readthedocs.io/en/latest/installation.html) page for more
details on installing JAX on different platforms with GPU or TPU support.

Note that on Google Colab, JAX support for GPU is enabled by default if the Colab backend
has GPU enabled. You simply need only install HSSM.

### Visualizing the model with `graphviz`

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

## Questions?

If you have any questions, please
[open an issue](https://github.com/lnccbrown/HSSM/issues) in our
[GitHub repo](https://github.com/lnccbrown/HSSM).
