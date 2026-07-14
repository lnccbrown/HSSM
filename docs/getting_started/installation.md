# Installation

**Important Update:** From HSSM 0.4.0, HSSM supports installation directly through `pip` or `uv` on all platforms. You can still install HSSM into conda environments via `pip`, but `conda install hssm` no longer installs the latest version of HSSM.

Please follow the instructions below to install HSSM.

## Option 1: Installation with uv

We highly recommend that the installation of HSSM is done through `uv`, a package manager that simplifies the installation of Python packages and their dependencies. You can install `uv` by following the instructions on the [uv official website](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, you can install uv in one of the following ways:

```bash
uv venv ## creates a new virtual environment
uv pip install hssm
```

You can also add hssm to an existing Python project by:

```bash
uv add hssm
```

## Option 2: Installation with pip

HSSM can also be installed directly through `pip`. You can install HSSM into any virtual environment via:

```bash
pip install hssm
```

### Install HSSM (with GPU Support)

To sample on an NVIDIA GPU, install HSSM with the CUDA extra matching your CUDA
version. This pulls in the GPU-enabled build of JAX for you:

```bash
pip install hssm[cuda12]  # CUDA 12
pip install hssm[cuda13]  # CUDA 13
```

!!! note

    JAX's CUDA wheels are Linux-only and require a compatible NVIDIA driver
    (>= 525 for CUDA 12, >= 580 for CUDA 13).

### Support for Apple Silicon, AMD, and other GPUs

JAX also has support other GPUs. Please follow the
[Official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
to install the correct version of JAX before installing HSSM.

### Install the dev version of HSSM

You can install the dev version of `hssm` directly from this repo:

```bash
pip install git+https://github.com/lnccbrown/HSSM.git
```

### Install HSSM on Google Colab

Google Colab comes with PyMC and JAX pre-configured. That holds true even if you
are using the GPU and TPU backend, so you simply need to install HSSM via pip on
Colab regardless of the backend you are using:

```bash
!pip install hssm
```

## Install optional dependencies

Whether you have installed HSSM via `pip`, `uv`, or GitHub, you might still need additional
packages installed for additional features such as sampling with `blackjax` or GPU support
for `JAX`. Please follow the instructions below if you need any of these additional
features:

### 1. Sampling with JAX through `numpyro`, `nutpie` or `blackjax`

JAX-based sampling is done through `numpyro`, `nutpie`, or `blackjax`. `numpyro` is installed as
a dependency by default. You need to have `blackjax` installed if you want to use the
`blackjax` sampler.

```bash
pip install blackjax nutpie
```

### 2. Visualizing the model with `graphviz`

Model graphs are created with `model.graph()` through `graphviz`. You need to have
the Graphviz binaries available on your `PATH` (the `dot` command) and then install
its Python binding:

#### Install the Graphviz binaries

Install Graphviz through your package manager (e.g. conda or Homebrew) or by
following the instructions on the
[graphviz official site](https://graphviz.org/download/) for your specific platform.
Make sure the `dot` command is on your `PATH`.

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
system to optimize its computation. You can follow
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
    a fresh virtual environment to prevent any problems with dependencies
    during the installation process. Please note that HSSM is only tested for Python 3.12
    through 3.14. Use unsupported Python versions with caution.

## Questions?

If you have any questions, please
[open an issue](https://github.com/lnccbrown/HSSM/issues) in our
[GitHub repo](https://github.com/lnccbrown/HSSM).
