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

### Run notebook automation locally

The notebook execution inventory is defined in the repo-level `notebooks.toml` file.
Use the shared notebook runner to validate the inventory, execute notebooks locally,
or build a report from a cluster run.

Install the notebook dependencies first:

```sh
uv sync --group dev --group notebook
```

Validate the notebook inventory without running anything:

```sh
uv run python scripts/notebook_jobs.py discover --include-disabled
```

Run all enabled notebooks locally and write a report under `.cache/notebook-runs/...`:

```sh
uv run python scripts/notebook_jobs.py run-all --fail-on-error
```

Run a single notebook locally into a specific artifact directory:

```sh
uv run python scripts/notebook_jobs.py run-all \
    --notebook docs/tutorials/plotting.ipynb \
    --run-dir .cache/notebook-runs/plotting
```

### Submit notebook runs to Slurm

You can submit the enabled notebooks as a Slurm job array in two ways:

- Use the Python helper to generate and submit `sbatch` directly.
- Use the checked-in Slurm submit helper `scripts/sbatch_submit_notebook_jobs.sh`,
    which prepares the run directory and submits the batch payload script with
    dynamic log paths.

If you want to inspect the generated `sbatch` command from the CLI helper, start with:

```sh
uv run python scripts/notebook_jobs.py submit-slurm \
    --partition cpu \
    --time-limit 08:00:00 \
    --mem 16G \
    --dry-run
```

If you want a real Slurm script workflow, use `scripts/sbatch_submit_notebook_jobs.sh`.
It prepares the run directory, captures the enabled indexes, and submits the batch
payload `scripts/submit_notebook_jobs.sh` with `--output` and `--error` under the
chosen run directory.

Recommended cluster workflow:

```sh
uv sync --group dev --group notebook
uv run python scripts/notebook_jobs.py discover --include-disabled
bash scripts/sbatch_submit_notebook_jobs.sh \
    --run-dir .cache/notebook-runs/cluster-$(date +%Y%m%dT%H%M%S) \
    --partition cpu \
    --time-limit 08:00:00 \
    --mem 16G
```

The checked-in payload script already includes default `#SBATCH` headers for job name,
partition, wall time, memory, CPU count, and fallback log files. The submit helper
overrides the log locations so they land under `RUN_DIR/submitted/`. Override resources
at submission time with helper flags, for example:

```sh
bash scripts/sbatch_submit_notebook_jobs.sh \
    --run-dir .cache/notebook-runs/cluster-$(date +%Y%m%dT%H%M%S) \
    --partition largemem \
    --time-limit 12:00:00 \
    --mem 32G \
    --cpus-per-task 4 \
    --export PYTENSOR_FLAGS='blas__ldflags=-L/usr/lib/x86_64-linux-gnu -lblas -llapack'
```

This writes a frozen notebook inventory, per-task logs, result JSON files, and the
generated submission metadata into the chosen run directory. The most useful paths are:

- `inventory.json`: snapshot of the notebook inventory for that run
- `results/`: one JSON result per notebook task
- `logs/`: one execution log per notebook task
- `submitted/submission.json`: the generated Slurm submission details
- `report.json` and `report.md`: aggregate report after running `aggregate`

If your cluster needs a BLAS/LAPACK environment, make sure that is configured before
submission, for example through your normal module setup or `.pytensorrc` configuration.
The CLI submit helper forwards `PYTENSOR_FLAGS` automatically. The shell submit helper
also forwards `PYTENSOR_FLAGS` when it is present in the environment, and you can pass
additional exports with repeated `--export` flags.

If you prefer the Python helper that submits directly, you can still pass additional
environment variables through repeated `--export` flags:

```sh
uv run python scripts/notebook_jobs.py submit-slurm \
    --partition cpu \
    --time-limit 08:00:00 \
    --mem 16G \
    --export PYTENSOR_FLAGS=blas__ldflags=-L/usr/lib/x86_64-linux-gnu -lblas -llapack \
    --export MY_CLUSTER_FLAG=value
```

You can also pass through scheduler-specific flags with the Python helper using
`--sbatch-arg`, for example:

```sh
uv run python scripts/notebook_jobs.py submit-slurm \
    --partition cpu \
    --time-limit 08:00:00 \
    --mem 16G \
    --sbatch-arg=--constraint=icelake \
    --sbatch-arg=--qos=normal
```

After the array completes, rebuild the final report if needed:

```sh
uv run python scripts/notebook_jobs.py aggregate \
    --run-dir .cache/notebook-runs/<run-id>
```

To inspect notebooks that need attention quickly:

```sh
cat .cache/notebook-runs/<run-id>/report.md
```

To rerun a single notebook after a failure, point `run-one` at the existing run directory
and use either the notebook path or the inventory index from `inventory.json`:

```sh
uv run python scripts/notebook_jobs.py run-one \
    --run-dir .cache/notebook-runs/<run-id> \
    --notebook docs/tutorials/plotting.ipynb \
    --fail-on-error
```

```sh
uv run python scripts/notebook_jobs.py run-one \
    --run-dir .cache/notebook-runs/<run-id> \
    --index 16 \
    --fail-on-error
```

If you want a fresh single-notebook run instead of reusing an existing run directory,
create a new one and target the notebook directly:

```sh
uv run python scripts/notebook_jobs.py run-all \
    --run-dir .cache/notebook-runs/plotting-rerun \
    --notebook docs/tutorials/plotting.ipynb \
    --fail-on-error
```
