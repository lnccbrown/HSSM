# Installation

## Install the HSSM package

### Install via `pip` from PyPI

HSSM is available through [PyPI](https://pypi.org/project/hssm/). The easiest way to
install HSSM is through pip.

```bash
pip install hssm
```

Because HSSM depends on very specific versions of PyMC, JAX and Bambi, we recommend that
you install HSSM into a dedicated virtual environment to avoid dependency conflicts.

### Install from GitHub

You can also install the bleeding edge version of `hssm` directly from
[our repo](https://github.com/lnccbrown/HSSM):

```
pip install git+https://github.com/lnccbrown/HSSM.git
```

## Install optional dependencies

Some functionalities in HSSM are available through optional dependencies.

### Sampling with JAX through `numpyro` or `blackjax`

JAX-based sampling is done through `numpyro` and `blackjax`. `numpyro` is installed as
a dependency by default. You need to have `blackjax` installed if you want to use the
`nuts_blackjax` sampler.

```bash
pip install blackjax
```

### Sampling with JAX support for GPU

The `nuts_numpyro` sampler uses JAX as the backend and thus can support sampling on nvidia
GPU. The only thing you need to do to take advantage of this is to install JAX with CUDA
support before installing HSSM. Here's one example:

```bash
python -m venv .venv  # Create a virtual environment
source .venv/bin/activate  # Activate the virtual environment

pip install --upgrade pip

# We need to limit the version of JAX for now due to some breaking
# changes introduced in JAX 0.4.16.
pip install --upgrade "jax[cuda11_pip]<0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install hssm
```

The example above shows how to install JAX with CUDA 11 support. Please refer to the
[JAX Installation](https://jax.readthedocs.io/en/latest/installation.html) page for more
details on installing JAX on different platforms with GPU or TPU support.

Note that on Google Colab, JAX support for GPU is enabled by default if the Colab backend
has GPU enabled. You simply need only install HSSM.

### Visualizing the model

Model graphs are created with `model.graph()` through `graphviz`. In order to use it,
you need to have `graphviz` installed system-wide and then install its Python binding:

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

1. `pip` installation fails with missing dependencies:

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
