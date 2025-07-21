# Setting up a local development environment

To contribute to HSSM, the first step is to clone [the HSSM repo](https://github.com/lnccbrown/HSSM) locally
and set up a local development environment. This guide will walk you through setting up the local
dev environment on different platforms.

## Step 1. Install uv

HSSM is managed by [uv](https://docs.astral.sh/uv/). uv is an extremely fast Python package and project
manager written in Rust that is gaining tremendous popularity in the Python world. If you have not
already installed it, you need to install uv following its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/). You can verify your uv installation by the following command.

```sh
uv --version
```

## Step 2. Clone the HSSM repo and set up the virtual environment

### Clone the HSSM repo

You can clone the HSSM repo with `git`:

```sh
git clone https://github.com/lnccbrown/HSSM.git
cd HSSM
```

### Set up the virtual environment and install dependencies

uv will handle the creation of the virtual environment with all dev and test
dependencies with the following command:

```sh
uv sync --group dev --group test
```

This tells uv to install not only the HSSM dependencies but also the `dev` and `test`
dependency groups defined in `pyproject.toml`.

### Optional: Set up JAX for GPU

If you need JAX to support GPU, you can install the GPU version of JAX through:

```sh
uv sync --all-groups --all-extras
```

or

```sh
uv sync --group dev --group test --extra cuda12
```

Please ensure that you have a GPU that supports CUDA 12 for this installation.

## Step 3. Set up a linear algebra package

Different from installation through `conda`, which is what we recommend for HSSM users,
when HSSM is installed through `uv` in a local development environment, `pytensor` does
not usually know how to find the linear algebra acceleration library on your system, and
when that happens, you will typically get slow inference speed using the default `pymc`
NUTS sampler with the following warning:

```
WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
```

If this happens, it means that we need to tell `pytensor` where the linear algebra
library is installed on your system. There are many ways to do this, but the most
convenient way is to create a `.pytensorrc` file in your home directory with the
following content:

```
[blas]
ldflags=...
```

What `...` actually is depends on your operating system and hardware architecture.
Here's the recommendation for certain setups:

### MacOS with ARM chips (M-series processors)

MacOS with ARM chips comes with
[accelerate framework](https://developer.apple.com/documentation/accelerate), and you
just need to tell your `pytensor` to use it:

```
[blas]
ldflags=-framework Accelerate
```

### Intel Macs, Linux and Windows (WSL) on Intel processors

Intel has the oneAPI Math Kernel Library (oneMKL) that you can use as your acceleration
library. You need to install the [oneMKL library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.now6fh)
and then point your `.pytensorrc` file to it:

```
[blas]
ldflags=-Lwherever/mkl/is -lmkl_core -lmkl_rt -lpthread -lm
```

Just replace `wherever/mkl/is` with the absolute path of MKL installed on your system.

!!! note

    If you wish to develop HSSM on Oscar, Brown University's HPC cluster, MKL is already
    available through a module. You can [follow this instruction](https://github.com/lnccbrown/HSSM/discussions/440)
    on how to write the `.pytensorrc` file.

### Other systems

On other systems, we recommend installing `openblas` or `lapack`. You can follow
the instructions on [the openblas website](http://www.openmathlib.org/OpenBLAS/) or
[the lapack website](https://www.netlib.org/lapack/) to install one of these libraries.

If you installed `openblas`, then your `.pytensorrc` file should look like this:

```
[blas]
ldflags=-L/path/to/openblas/lib -lopenblas
```

If you installed `openblas`, then your `.pytensorrc` file should look like this:

```
[blas]
ldflags=-L/path/to/lapack/lib -lblas
```

## Follow-up

Once the above steps are done, you should be able to run a `pytensor` based sampler
without any warnings.

We recommend that you also configure your IDE to use the Python interpreter in the
virtual environment created by `uv`. Typically, this virtual environment is in the
`.venv` directory under the `HSSM` directory. If you use Visual Studio Code, it will
automatically detect this virtual environment and ask if you want to use the Python
interpreter in that virtual environment as the interpreter.

### Run test suite

We recommend that you run the test suite with `uv run` command:

```sh
uv run pytest
```
