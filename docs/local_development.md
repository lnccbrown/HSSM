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
uv sync --group notebook # dev group is synced by default. notebook groups is required for running the Jupyter notebook in the dev environment.
```

This tells uv to install not only the HSSM dependencies but also the `dev` and `test`
dependency groups defined in `pyproject.toml`.

### Optional: Set up JAX for GPU

If you need JAX to support GPU, choose exactly one CUDA extra (the `cuda12` and
`cuda13` extras are mutually exclusive, so do not use `--all-extras`):

```sh
uv sync --group dev --extra cuda12
```

or, for CUDA 13:

```sh
uv sync --group dev --extra cuda13
```

Please ensure that you have a GPU that supports CUDA 12 or CUDA 13,
respectively, for this installation.

## Follow-up

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
