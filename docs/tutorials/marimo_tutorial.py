import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.notebook_dir()
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <center><img src="public/HSSM_logo.png"></center>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This tutorial provides a comprehensive introduction to the HSSM package for Hierarchical Bayesian Estimation of Sequential Sampling Models.

    To make the most of the tutorial, let us cover the functionality of the key supporting packages that we use along the way.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Colab Instructions

    If you would like to run this tutorial on Google colab, please click this [link](https://github.com/lnccbrown/HSSM/blob/main/docs/tutorials/main_tutorial.ipynb).

    Once you are *in the colab*, follow the *installation instructions below* and then **restart your runtime**.

    Just **uncomment the code in the next code cell** and run it!

    **NOTE**:

    You may want to *switch your runtime* to have a GPU or TPU. To do so, go to *Runtime* > *Change runtime type* and select the desired hardware accelerator.

    Note that if you switch your runtime you have to follow the installation instructions again.
    """)
    return


@app.cell
def _():
    # If running this on Colab, please uncomment the next line
    # !pip install git+https://github.com/lnccbrown/HSSM@workshop_tutorial
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic Imports
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    # Basics
    import numpy as np
    from matplotlib import pyplot as plt

    random_seed_sim = 134
    np.random.seed(random_seed_sim)
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Simulation

    We will rely on the [ssms](https://github.com/AlexanderFengler/ssm-simulators) package for data simulation repeatedly. Let's look at a basic isolated use case below.

    As an example, let's use [ssms](https://github.com/AlexanderFengler/ssm-simulators) to simulate from the basic [Drift Diffusion Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2474742/) (a running example in this tutorial).

    <center><img src="public/DDM_with_params_pic.png"></center>

    If you are not familiar with the DDM. For now just consider that it has four parameters.

    - `v` the drift rate
    - `a` the boundary separation
    - `t` the non-decision time
    - `z` the a priori decision bias (starting point)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using `simulate_data()`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM comes with a basic simulator function supplied the `simulate_data()` function. We can use this function to create synthetic datasets.

    Below we show the most basic usecase:

    We wish to generate `500` datapoints (trials) from the standard [Drift Diffusion Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2474742/) with a fixed parameters, `v = 0.5, a = 1.5, z = 0.5, t = 0.5`.


    **Note**:

    In the course of the tutorial, we will see multiple strategies for synthetic dataset generation, this being the most straightforward one.
    """)
    return


@app.cell
def _():
    # Single dataset
    import arviz as az  # Visualization
    import bambi as bmb  # Model construction
    import hddm_wfpt
    import jax
    import pytensor  # Graph-based tensor library

    import hssm

    # pytensor.config.floatX = "float32"
    # jax.config.update("jax_enable_x64", False)

    v_true = 0.5
    a_true = 1.5
    z_true = 0.5
    t_true = 0.5

    # Call the simulator function
    dataset_simulated = hssm.simulate_data(
        model="ddm", theta=dict(v=v_true, a=a_true, z=z_true, t=t_true), size=500
    )

    dataset_simulated
    return (
        a_true,
        az,
        bmb,
        dataset_simulated,
        hddm_wfpt,
        hssm,
        jax,
        t_true,
        v_true,
        z_true,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If instead you wish to supply a parameter that *varies by trial* (a lot more on this later), you can simply supply a vector of parameters to the `theta` dictionary, when calling the simulator.

    **Note**:

    The `size` argument conceptually functions as *number of synthetic datasets*. So if you supply a parameter as a `(1000,)` vector, then the simulator assumes that one dataset consists of `1000` trials, hence if we set the `size = 1` as below, we expect in return a dataset with `1000` trials.
    """)
    return


@app.cell
def _(hssm, np, t_true, v_true, z_true):
    # a changes trial wise
    a_trialwise = np.random.normal(loc=2, scale=0.3, size=1000)

    dataset_a_trialwise = hssm.simulate_data(
        model="ddm",
        theta=dict(
            v=v_true,
            a=a_trialwise,
            z=z_true,
            t=t_true,
        ),
        size=1,
    )

    dataset_a_trialwise
    return (a_trialwise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we wish to simulate from another model, we can do so by changing the `model` string.

    The number of models we can simulate differs from the number of models for which we have likelihoods available (both will increase over time). To get the models for which likelihood functions are supplied out of the box, we should inspect  `hssm.HSSM.supported_models`.
    """)
    return


@app.cell
def _(hssm):
    hssm.HSSM.supported_models
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we wish to check more detailed information about a given supported model, we can use the accessor `get_default_model_config` under `hssm.modelconfig`. For example, we inspect `ddm` model configuration below.
    """)
    return


@app.cell
def _(hssm):
    hssm.modelconfig.get_default_model_config("ddm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This dictionary contains quite a bit of information. For purposes of *simulating data from a given model*, we will highlight two aspects:

    1.  The key `list_of_params` provides us with the necessary information to define out `theta` dictionary
    2.  The `bounds` key inside the `likelihoods` sub-dictionaries, provides us with an indication of reasonable parameter values.

    The `likelihoods` dictionary inhabits three sub-directories for the `ddm` model, since we have all three, an `analytical`, an `approx_differentiable` (LAN) and a `blackbox` likelihood available. For many models, we will be able to access only one or two types of likelihoods.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using `ssm-simulators`

    Internally, HSSM natively makes use of the [ssm-simulators](https://github.com/AlexanderFengler/ssm-simulators) package for forward simulation of models.
    `hssm.simulate_data()` functions essentially as a convenience-wrapper.

    Below we illustrate how to simulate data using the `ssm-simulators` package directly, to generate an equivalent dataset as created above. We will use the *third* way of passing parameters to the simulator, which is as a parameter-*matrix*.

    **Notes**:

    1. If you pass parameters as a parameter matrix, make sure the column ordering is correct. You can follow the parameter ordering under `hssm.defaults.default_model_config['ddm']['list_params']`.

    2. This is a minimal example, for more information about the package, check the associated [github-page](https://github.com/AlexanderFengler/ssm-simulators).
    """)
    return


@app.cell
def _(a_trialwise, np, t_true, v_true, z_true):
    import pandas as pd
    from ssms.basic_simulators.simulator import simulator

    # a changes trial wise
    theta_mat = np.zeros((1000, 4))
    theta_mat[:, 0] = v_true  # v
    theta_mat[:, 1] = a_trialwise  # a
    theta_mat[:, 2] = z_true  # z
    theta_mat[:, 3] = t_true  # t

    # simulate data
    sim_out_trialwise = simulator(
        theta=theta_mat,  # parameter_matrix
        model="ddm",  # specify model (many are included in ssms)
        n_samples=1,  # number of samples for each set of parameters
        # (plays the role of `size` parameter in `hssm.simulate_data`)
    )

    # Turn into nice dataset
    dataset_trialwise = pd.DataFrame(
        np.column_stack(
            [sim_out_trialwise["rts"][:, 0], sim_out_trialwise["choices"][:, 0]]
        ),
        columns=["rt", "response"],
    )

    dataset_trialwise
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will stick to `hssm.simulate_data()` in this tutorial, to keep things simple.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ArviZ for Plotting

    <center><img src="public/arviz.png"></center>


    We use the [ArviZ](https://python.arviz.org/en/stable/) package for most of our plotting needs.
    ArviZ is a useful aid for plotting when doing anything Bayesian.

    It works with HSSM out of the box, by virtue of HSSMs reliance on [PyMC](https://www.pymc.io/welcome.html) for model construction and sampling.

    Checking out the [ArviZ Documentation](https://python.arviz.org/en/stable/getting_started/index.html) is a good idea to give you communication superpowers for not only your HSSM results, but also other libraries in the Bayesian Toolkit such as [NumPyro](https://num.pyro.ai/en/latest/index.html#introductory-tutorials) or [STAN](https://mc-stan.org/users/documentation/).

    We will see [ArviZ](https://python.arviz.org/en/stable/) plots throughout the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Main Tutorial
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's proceed to simulate a simple dataset for our first example.
    """)
    return


@app.cell
def _(a_true, hssm, t_true, v_true, z_true):
    # Specify
    param_dict_init = dict(
        v=v_true,
        a=a_true,
        z=z_true,
        t=t_true,
    )


    dataset_simulated_main = hssm.simulate_data(
        model="ddm",
        theta=param_dict_init,
        size=500,
    )

    dataset_simulated_main
    return dataset_simulated_main, param_dict_init


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First HSSM Model

    In this example we will use the *analytical likelihood function* computed as suggested in [this paper](https://psycnet.apa.org/record/2009-11068-003).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instantiate the model

    To instantiate our `HSSM` class, in the simplest version, we only need to provide an appropriate dataset.
    The dataset is expected to be a `pandas.DataFrame` with at least two columns, respectively called `rt` (for reaction time) and `response`.
    Our data simulated above is already in the correct format, so let us try to construct the class.

    **NOTE:**

    If you are a user of the [HDDM](https://github.com/hddm-devs/hddm) python package, this workflow should seem very familiar.
    """)
    return


@app.cell
def _(dataset_simulated_main, hssm):
    simple_ddm_model = hssm.HSSM(data=dataset_simulated_main)
    return (simple_ddm_model,)


@app.cell
def _(simple_ddm_model):
    simple_ddm_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `print()` function gives us some basic information about our model including the *number of observations* the *parameters in the model* and their respective *prior setting*. We can also create a nice little graphical representation of our model...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Graph

    Since `HSSM` creates a `PyMC Model`, we can can use the `.graph()` function, to get a graphical representation of the the model we created.
    """)
    return


@app.cell
def _(mo, simple_ddm_model):
    graph_simulated_main = simple_ddm_model.graph()
    # For marimo display
    png_bytes_main = graph_simulated_main.pipe(format="png")
    mo.image(png_bytes_main)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the simplest model we can build. The graph above follows **plate notation**, commonly used for **probabilistic graphical models**.

    - We have our basic parameters (unobserved, **white nodes**), these are *random variables* in the model and we want to estimate them
    - Our observed reaction times and choices (`SSMRandomVariable`, **grey node**), are fixed (or conditioned on).
    - **Rounded rectangles** provide us with information about dimensionality of objects
    - **Rectangles with sharp edges** represent *deterministic*, but *computed* quantities (not shown here, but in later models)

    This notation is helpful to get a quick overview of the structure of a given model we construct.

    The `graph()` function of course becomes a lot more interesting and useful for more complicated models!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sample from the Model

    We can now call the `.sample()` function, to get posterior samples. The main arguments you may want to change are listed in the function call below.

    Importantly, multiple backends are possible. We choose the `nuts_numpyro` backend below,
    which in turn compiles the model to a [`JAX`](https://github.com/google/jax) function.
    """)
    return


@app.cell
def _(simple_ddm_model):
    infer_data_simple_ddm_model = simple_ddm_model.sample(
        sampler="mcmc",  # type of sampler to choose, 'nuts_numpyro',
        # 'nuts_blackjax' of default pymc nuts sampler
        cores=1,  # how many cores to use
        chains=2,  # how many chains to run
        draws=500,  # number of draws from the markov chain
        tune=1000,  # number of burn-in samples
        idata_kwargs=dict(log_likelihood=True),  # return log likelihood
        mp_ctx="spawn",
    )  # mp_ctx="forkserver")
    return (infer_data_simple_ddm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We sampled from the model, let's look at the output...
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    type(infer_data_simple_ddm_model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Errr... a closer look might be needed here!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference Data / What gets returned from the sampler?

    The sampler returns an [ArviZ](https://python.arviz.org/en/stable/) `InferenceData` object.

    To understand all the logic behind these objects and how they mesh with the Bayesian Workflow, we refer you to the [ArviZ Documentation](https://python.arviz.org/en/stable/getting_started/index.html).

    `InferenceData` is build on top of [xarrays](https://docs.xarray.dev/en/stable/index.html). The [xarray documentation](https://docs.xarray.dev/en/stable/index.html) will help you understand in more detail how to manipulate these objects.

    But let's take a quick high-level look to understand roughly what we are dealing with here!
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    infer_data_simple_ddm_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that in our case, `infer_data_simple_ddm_model` contains four basic types of data (note: this is extensible!)

    - `posterior`
    - `log_likelihood`
    - `sample_stats`
    - `observed_data`

    The `posterior` object contains our traces for each of the parameters in the model. The `log_likelihood` field contains the trial wise log-likelihoods for each sample from the posterior. The `sample_stats` field contains information about the sampler run. This can be important for chain diagnostics, but we will not dwell on this here. Finally we retreive our `observed_data`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic Manipulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Accessing groups and variables
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    infer_data_simple_ddm_model.posterior
    return


@app.cell
def _(infer_data_simple_ddm_model):
    infer_data_simple_ddm_model.posterior.a.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To simply access the underlying data as a `numpy.ndarray`, we can use `.values` (as e.g. when using `pandas.DataFrame` objects).
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    type(infer_data_simple_ddm_model.posterior.a.values)
    return


@app.cell
def _():
    # infer_data_simple_ddm_model.posterior.a.values
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Combine `chain` and `draw` dimension

    When operating directly on the `xarray`, you will often find it useful to collapse the `chain` and `draw` coordinates into a single coordinate.
    **Arviz** makes this easy via the `extract` method.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model):
    idata_extracted = az.extract(infer_data_simple_ddm_model)
    idata_extracted
    return (idata_extracted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since *Arviz* really just calls the `.stack()` method from *xarray*, here the corresponding example using the lower level `xarray` interface.
    """)
    return


@app.cell
def _(infer_data_simple_ddm_model):
    infer_data_simple_ddm_model.posterior.stack(sample=("chain", "draw"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Making use of ArviZ

    Working with the `InferenceData` directly, is very helpful if you want to include custom computations into your workflow.
    For a basic Bayesian Workflow however, you will often find that standard functionality available through [ArviZ](https://python.arviz.org/en/stable/)
    suffices.

    Below we provide a few examples of useful **Arviz** outputs, which come handy for analyzing your traces (MCMC samples).

    #### Summary table

    Let's take a look at a summary table for our posterior.
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model, simple_ddm_model):
    az.summary(
        infer_data_simple_ddm_model,
        var_names=[var_name.name for var_name in simple_ddm_model.pymc_model.free_RVs],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This table returns the parameter-wise mean of our posterior and a few extra statistics.

    Of these extra statistics, the one-stop shop for flagging convergence issues is the `r_hat` value, which
    is reported in the right-most column.

    To navigate this statistic, here is a rule of thumb widely used in applied Bayesian statistics.

    If you find an `r_hat` value `> 1.01`, it warrants investigation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trace plot
    """)
    return


@app.cell
def _(az, infer_data_simple_ddm_model, mo, param_dict_init, plt):
    ax_trace = az.plot_trace(
        infer_data_simple_ddm_model,  # we exclude the log_likelihood traces here
        lines=[(key_, {}, param_dict_init[key_]) for key_ in param_dict_init],
    )

    fig_trace = plt.gcf()

    plt.tight_layout()
    mo.mpl.interactive(fig_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `.sample()` function also sets a `trace` attribute, on our `hssm` class, so instead, we could call the plot like so:
    """)
    return


@app.cell
def _(az, mo, param_dict_init, plt, simple_ddm_model):
    az.plot_trace(
        simple_ddm_model.traces,
        lines=[(key_, {}, param_dict_init[key_]) for key_ in param_dict_init],
    );

    fig_trace2 = plt.gcf()

    plt.tight_layout()
    mo.mpl.interactive(fig_trace2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this tutorial we are most often going to use the latter way of accessing the traces, but there is no preferred option.

    Let's look at a few more plots.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Forest Plot

    The forest plot is commonly used for a quick visual check of the marginal posteriors. It is very effective for intuitive communication of results.
    """)
    return


@app.cell
def _(az, simple_ddm_model):
    az.plot_forest(simple_ddm_model.traces)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### Combining Chains
    By default, chains are separated out into *separate caterpillars*, however
    sometimes, especially if you are looking at a forest plot which includes many posterior parameters at once, you want to declutter and collapse the chains into single caterpillars.
    In this case you can `combine` chains instead.
    """)
    return


@app.cell
def _(az, simple_ddm_model):
    az.plot_forest(simple_ddm_model.traces, combined=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Basic Marginal Posterior Plot

    Another way to view the marginal posteriors is provided by the `plot_posterior()` function. It shows the mean and by default the $94\%$ HDIs.
    """)
    return


@app.cell
def _(az, simple_ddm_model):
    az.plot_posterior(simple_ddm_model.traces)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Especially for parameter recovery studies, you may want to include **reference values** for the parameters of interest.

    You can do so using the `ref_val` argument. See the example below:
    """)
    return


@app.cell
def _(az, param_dict_init, simple_ddm_model):
    az.plot_posterior(
        simple_ddm_model.traces,
        ref_val=[
            param_dict_init[var_name]
            for var_name in simple_ddm_model.traces.posterior.data_vars
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since it is sometimes useful, especially for more complex cases, below an alternative approach in which we pass `ref_val` as a dictionary.
    """)
    return


@app.cell
def _(az, param_dict_init, simple_ddm_model):
    az.plot_posterior(
        simple_ddm_model.traces,
        ref_val={
            "v": [{"ref_val": param_dict_init["v"]}],
            "a": [{"ref_val": param_dict_init["a"]}],
            "z": [{"ref_val": param_dict_init["z"]}],
            "t": [{"ref_val": param_dict_init["t"]}],
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Posterior Pair Plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The posterior pair plot show us bi-variate traceplots and is useful to check for simple parameter tradeoffs that may emerge. The simplest (linear) tradeoff may be a high correlation between two parameters.
    This can be very helpful in diagnosing sampler issues for example. If such tradeoffs exist, one often see extremely *wide marginal distributions*.

    In our `ddm` example, we see a little bit of a tradeoff between `a` and `t`, as well as between `v` and `z`, however nothing concerning.
    """)
    return


@app.cell
def _(az, mo, param_dict_init, plt, simple_ddm_model):
    az.plot_pair(
        simple_ddm_model.traces,
        kind="kde",
        reference_values=param_dict_init,
        marginals=True,
    );

    pair_plot = plt.gcf()

    plt.tight_layout()
    mo.mpl.interactive(pair_plot)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The few plot we showed here are just the beginning: [ArviZ](https://python.arviz.org/en/stable/) has a much broader spectrum of graphs and other convenience function available. Just check the [documentation](https://python.arviz.org/en/stable/).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute Quantities from idata

    #### Example: Mean and Covariance of Posterior Parameters

    As a simple example, let us calculate the covariance matrix for our posterior samples.
    """)
    return


@app.cell
def _(idata_extracted, np, plt):
    # Calculate the correlation matrix
    posterior_correlation_matrix = np.corrcoef(
        np.stack(
            [idata_extracted[var_].values for var_ in idata_extracted.data_vars.variables]
        )
    )
    num_vars = posterior_correlation_matrix.shape[0]

    # Make heatmap
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(posterior_correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)
    ax.set_title("Posterior Correlation Matrix")

    # Add ticks
    ax.set_xticks(range(posterior_correlation_matrix.shape[0]))
    ax.set_xticklabels([var_ for var_ in idata_extracted.data_vars.variables])
    ax.set_yticks(range(posterior_correlation_matrix.shape[0]))
    ax.set_yticklabels([var_ for var_ in idata_extracted.data_vars.variables])

    # Annotate heatmap
    for i_cor in range(num_vars):
        for j_cor in range(num_vars):
            ax.text(
                j_cor,
                i_cor,
                f"{posterior_correlation_matrix[i_cor, j_cor]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HSSM Model based on LAN likelihood

    With HSSM you can switch between pre-supplied models with a simple change of argument. The type of likelihood that will be accessed might change in the background for you.

    Here we see an example in which the underlying likelihood is now a [LAN](https://elifesciences.org/articles/65074).

    We will talk more about different types of likelihood functions and backends later in the tutorial. For now just keep the following in mind:

    There are three types of likelihoods

    1. `analytic`
    2. `approx_differentiable`
    3. `blackbox`

    To check which type is used in your HSSM model simple type:
    """)
    return


@app.cell
def _(simple_ddm_model):
    simple_ddm_model.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ah... we were using an `analytical` likelihood with the DDM model in the last section.
    Now let's see something different!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulating Angle Data

    Again, let us simulate a simple dataset. This time we will use the `angle` model (passed via the `model` argument to the `simulator()` function).


    This model is distinguished from the basic `ddm` model by an additional `theta` parameter which specifies the angle with which the decision boundaries collapse over time.

    <center><img src="public/ANGLE_with_params_pic.png"></center>

    DDMs with collapsing bounds have been of significant interest in the theoretical literature, but applications were rare due to a lack of analytical likelihoods. HSSM facilitates inference with such models via the our `approx_differentiable` likelihoods. HSSM ships with a few predefined models based on [LANs](https://elifesciences.org/articles/65074), but really we don't want to overemphasize those. They reflect the research interest of our and adjacent labs to a great extend.

    Instead, we encourage the community to contribute to this model reservoir (more on this later).
    """)
    return


@app.cell
def _(hssm):
    # Simulate angle data
    v_angle_true = 0.5
    a_angle_true = 1.5
    z_angle_true = 0.5
    t_angle_true = 0.2
    theta_angle_true = 0.2

    param_dict_angle = dict(v=0.5, a=1.5, z=0.5, t=0.2, theta=0.2)

    lines_list_angle = [(key_, {}, param_dict_angle[key_]) for key_ in param_dict_angle]

    dataset_angle = hssm.simulate_data(model="angle", theta=param_dict_angle, size=1000)
    return dataset_angle, lines_list_angle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We pass a single additional argument to our `HSSM` class and set `model='angle'`.
    """)
    return


@app.cell
def _(dataset_angle, hssm):
    model_angle = hssm.HSSM(data=dataset_angle, model="angle")

    model_angle
    return (model_angle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model graph now show us an additional parameter `theta`!
    """)
    return


@app.cell
def _(mo, model_angle):
    model_angle_graph = model_angle.graph()
    png_bytes_angle = model_angle_graph.pipe(format="png")
    mo.image(png_bytes_angle)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the type of likelihood that is used under the hood ...
    """)
    return


@app.cell
def _(model_angle):
    model_angle.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok so here we rely on a likelihood of the `approx_differentiable` kind.

    As discussed, with the initial set of pre-supplied likelihoods, this implies that we are using a [LAN](https://elifesciences.org/articles/65074) in the background.
    """)
    return


@app.cell
def _(jax, model_angle):
    jax.config.update("jax_enable_x64", False)
    infer_data_angle = model_angle.sample(
        sampler="nuts_numpyro",
        chains=2,
        cores=2,
        draws=500,
        tune=500,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        mp_ctx="spawn",
    )
    return


@app.cell
def _(az, lines_list_angle, mo, model_angle, plt):
    az.plot_trace(model_angle.traces, lines=lines_list_angle)
    plt.tight_layout()

    plot_trace3 = plt.gcf()

    plt.tight_layout()
    mo.mpl.interactive(plot_trace3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing Priors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    HSSM allows you to specify priors quite freely. If you used HDDM previously, you may feel relieved to read that your hands are now untied!


    <center><img src="https://media.giphy.com/media/99uUPYHtmaJS8/giphy.gif"></center>

    With HSSM we have multiple routes to priors. But let's first consider a special case:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fixing a parameter to a given value
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assume that instead of fitting all parameters of the DDM,

    <center><img src="public/DDM_with_params_pic.png"></center>

    we instead want to fit only the `v` (drift) parameter, setting all other parameters to fixed scalar values.

    <center><img src="public/DDM_only_v_pic.png"></center>

    HSSM makes this extremely easy!
    """)
    return


@app.cell
def _(param_dict_init):
    param_dict_init
    return


@app.cell
def _(dataset_simulated_main, hssm, param_dict_init):
    ddm_model_only_v = hssm.HSSM(
        data=dataset_simulated_main,
        model="ddm",
        a=param_dict_init["a"],
        t=param_dict_init["t"],
        z=param_dict_init["z"],
    )
    return (ddm_model_only_v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since we fix all but one parameter, we therefore estimate only one parameter. This should be reflected in our model graph, where we expect only one free random variable `v`:
    """)
    return


@app.cell
def _(ddm_model_only_v, mo):
    ddm_model_only_v_graph = ddm_model_only_v.graph()
    ddm_model_only_v_graph_png = ddm_model_only_v_graph.pipe(format="png")
    mo.image(ddm_model_only_v_graph_png)
    return


@app.cell
def _(ddm_model_only_v):
    ddm_model_only_v.sample(
        sampler="mcmc",
        chains=2,
        cores=2,
        draws=500,
        tune=500,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        mp_ctx="spawn",
    )
    return


@app.cell
def _(az, ddm_model_only_v, param_dict_init, plt):
    az.plot_trace(
        ddm_model_only_v.traces.posterior, lines=[("v", {}, param_dict_init["v"])]
    );
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of the trace on the right, a useful alternative / complement is the **rank plot**.
    As a rule of thumb, if the rank plots within chains look *uniformly distributed*, then our chains generally exhibit *good mixing*.
    """)
    return


@app.cell
def _(az, ddm_model_only_v):
    az.plot_trace(ddm_model_only_v.traces, kind="rank_bars")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Named priors

    We can choose any [PyMC](https://www.pymc.io/welcome.html) `Distribution` to specify a prior for a given parameter.

    Even better, if natural *parameter bounds* are provided, HSSM *automatically truncates the prior distribution* so that it respect these bounds.

    Below is an example in which we specify a *Normal* prior on the `v` parameter of the DDM.

    We choose a *ridiculously low* $\sigma$ value, to illustrate it's regularizing effect on the parameter (just so we see a difference and you are convinced that something changed).
    """)
    return


@app.cell
def _(dataset_simulated_main, hssm):
    model_normal = hssm.HSSM(
        data=dataset_simulated_main,
        include=[
            {
                "name": "v",
                "prior": {"name": "Normal", "mu": 0, "sigma": 0.01},
            }
        ],
    )
    return (model_normal,)


@app.cell
def _(model_normal):
    model_normal
    return


@app.cell
def _(model_normal):
    infer_data_normal = model_normal.sample(
        sampler="mcmc",
        chains=2,
        cores=2,
        draws=500,
        tune=500,
        idata_kwargs=dict(log_likelihood=False),  # no need to return likelihoods here
        mp_ctx="spawn",
    )
    return


@app.cell
def _(az, model_normal, param_dict_init):
    az.plot_trace(
        model_normal.traces,
        lines=[(key_, {}, param_dict_init[key_]) for key_ in param_dict_init],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Observe how we reused our previous dataset with underlying parameters

    - `v = 0.5`
    - `a = 1.5`
    - `z = 0.5`
    - `t = 0.2`

    In contrast to our previous sampler round, in which we used Uniform priors, here the `v` estimate is shrunk severley towared $0$ and the `t` and `z` parameter estimates are very biased to make up for this distortion. Also, overall we see a lot of divergences now, which is a sign of poor sampler performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HSSM Model with Regression


    <center><img src="public/bambi.png"></center>

    Crucial to the scope of HSSM is the ability to link parameters with trial-by-trial covariates via (hierarchical, but more on this later) general linear models.

    In this section we explore how HSSM deals with these models. No big surprise here... it's simple!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 1: One parameter is a Regression Target
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Simulating Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first simulate some data, where the trial-by-trial parameters of the `v` parameter in our model are driven by a simple linear regression model.


    The regression model is driven by two (random) covariates `x` and `y`, respectively with coefficients of $0.8$ and $0.3$ which are also simulated below.
    We set the intercept to $0.3$.

    The rest of the parameters are fixed to single values as before.
    """)
    return


@app.cell
def _(hssm, np):
    # Set up trial by trial parameters
    v_intercept = 0.3
    x = np.random.uniform(-1, 1, size=1000)
    v_x = 0.8
    y = np.random.uniform(-1, 1, size=1000)
    v_y = 0.3
    v_reg_v = v_intercept + (v_x * x) + (v_y * y)

    # rest
    a_reg_v = 1.5
    z_reg_v = 0.5
    t_reg_v = 0.1

    param_dict_reg_v = dict(
        a=1.5, z=0.5, t=0.1, v=v_reg_v, v_x=v_x, v_y=v_y, v_Intercept=v_intercept, theta=0.0
    )

    # base dataset
    dataset_reg_v = hssm.simulate_data(model="ddm", theta=param_dict_reg_v, size=1)

    # Adding covariates into the datsaframe
    dataset_reg_v["x"] = x
    dataset_reg_v["y"] = y
    return dataset_reg_v, param_dict_reg_v, v_intercept, v_x, v_y, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Basic Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now create the `HSSM` model.

    Notice how we set the `include` argument. The include argument expects a list of dictionaries, one dictionary for each parameter to be specified via a regression model.

    Four `keys` are expected to be set:

    - The `name` of the parameter,
    - Potentially a `prior` for each of the regression level parameters ($\beta$'s),
    - The regression `formula`
    - A `link` function.

    The regression formula follows the syntax in the [formulae](https://pypi.org/project/formulae/) python package (as used by the [Bambi](https://bambinos.github.io/bambi/) package for building Bayesian Hierarchical Regression Models.

    [Bambi](https://bambinos.github.io/bambi/) forms the main model-construction backend of HSSM.
    """)
    return


@app.cell
def _(dataset_reg_v, hssm):
    model_reg_v_simple = hssm.HSSM(
        data=dataset_reg_v, include=[{"name": "v", "formula": "v ~ 1 + x + y"}]
    )
    return (model_reg_v_simple,)


@app.cell
def _(model_reg_v_simple):
    model_reg_v_simple
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### `Param` class
    As illustrated below, there is an alternative way of specifying the parameter specific data via the `Param` class.
    """)
    return


@app.cell
def _(dataset_reg_v, hssm):
    model_reg_v_simple_new = hssm.HSSM(
        data=dataset_reg_v, include=[hssm.Param(name="v", formula="v ~ 1 + x + y")]
    )
    return (model_reg_v_simple_new,)


@app.cell
def _(model_reg_v_simple_new):
    model_reg_v_simple_new
    return


@app.cell
def _(mo, model_reg_v_simple):
    model_reg_v_simple_graph = model_reg_v_simple.graph()

    model_reg_v_simple_png = model_reg_v_simple_graph.pipe(format="png")
    mo.image(model_reg_v_simple_png)
    return


@app.cell
def _(model_reg_v_simple):
    print(model_reg_v_simple)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Custom Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These were the defaults, with a little extra labor, we can e.g. customize the choice of priors for each parameter in the model.
    """)
    return


@app.cell
def _(dataset_reg_v, hssm):
    model_reg_v = hssm.HSSM(
        data=dataset_reg_v,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
                "link": "identity",
            }
        ],
    )
    return (model_reg_v,)


@app.cell
def _(model_reg_v):
    model_reg_v
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how `v` is now set as a regression.
    """)
    return


@app.cell
def _(model_reg_v):
    infer_data_reg_v = model_reg_v.sample(
        sampler="mcmc",
        chains=2,
        cores=2,
        draws=500,
        tune=500,
        mp_ctx="spawn",
    )
    return (infer_data_reg_v,)


@app.cell
def _(infer_data_reg_v):
    infer_data_reg_v
    return


@app.cell
def _():
    # az.plot_forest(model_reg_v.traces)
    return


@app.cell
def _(az, model_reg_v, param_dict_reg_v):
    az.plot_trace(
        model_reg_v.traces,
        var_names=["~v"],
        lines=[(key_, {}, param_dict_reg_v[key_]) for key_ in param_dict_reg_v],
    )
    return


@app.cell
def _(az, model_reg_v, param_dict_reg_v, plt):
    az.plot_trace(
        model_reg_v.traces,
        var_names=["~v"],
        lines=[(key_, {}, param_dict_reg_v[key_]) for key_ in param_dict_reg_v],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(az, model_reg_v):
    # Looks like parameter recovery was successful
    az.summary(model_reg_v.traces, var_names=["~v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 2: One parameter is a Regression (LAN)

    We can do the same thing with the `angle` model.

    **Note**:

    Our dataset was generated from the basic DDM here, so since the DDM assumes stable bounds, we expect the `theta` (angle of linear collapse) parameter to be recovered as close to $0$.
    """)
    return


@app.cell
def _(dataset_reg_v, hssm):
    model_reg_v_angle = hssm.HSSM(
        data=dataset_reg_v,
        model="angle",
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {
                        "name": "Uniform",
                        "lower": -3.0,
                        "upper": 3.0,
                    },
                    "x": {
                        "name": "Uniform",
                        "lower": -1.0,
                        "upper": 1.0,
                    },
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
                "link": "identity",
            }
        ],
    )
    return (model_reg_v_angle,)


@app.cell
def _(mo, model_reg_v_angle):
    model_reg_v_angle_graph = model_reg_v_angle.graph()

    model_reg_v_angle_png = model_reg_v_angle_graph.pipe(format="png")
    mo.image(model_reg_v_angle_png)
    return


@app.cell
def _(model_reg_v_angle):
    trace_reg_v_angle = model_reg_v_angle.sample(
        sampler="mcmc",
        chains=1,
        cores=1,
        draws=1000,
        tune=500,
        mp_ctx="spawn",
    )
    return


@app.cell
def _(az, model_reg_v_angle, param_dict_reg_v, plt):
    az.plot_trace(
        model_reg_v_angle.traces,
        var_names=["~v"],
        lines=[(key_, {}, param_dict_reg_v[key_]) for key_ in param_dict_reg_v],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great! `theta` is recovered correctly, on top of that, we have reasonable recovery for all other parameters!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 3: Multiple Parameters are Regression Targets (LAN)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's get a bit more ambitious. We may, for example, want to try a regression on a few of our basic model parameters at once. Below we show an example where we model both the `a` and the `v` parameters with a regression.

    **NOTE:**

    In our dataset of this section, only `v` is *actually* driven by a trial-by-trial regression, so we expect the regression coefficients for `a` to hover around $0$ in our posterior.
    """)
    return


@app.cell
def _(dataset_reg_v, hssm, param_dict_reg_v):
    # Instantiate our hssm model
    from copy import deepcopy

    param_dict_reg_v_a = deepcopy(param_dict_reg_v)
    param_dict_reg_v_a["a_Intercept"] = param_dict_reg_v_a["a"]
    param_dict_reg_v_a["a_x"] = 0
    param_dict_reg_v_a["a_y"] = 0

    hssm_reg_v_a_angle = hssm.HSSM(
        data=dataset_reg_v,
        model="angle",
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": -3.0, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "v ~ 1 + x + y",
            },
            {
                "name": "a",
                "prior": {
                    "Intercept": {"name": "Uniform", "lower": 0.5, "upper": 3.0},
                    "x": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                    "y": {"name": "Uniform", "lower": -1.0, "upper": 1.0},
                },
                "formula": "a ~ 1 + x + y",
            },
        ],
    )
    return hssm_reg_v_a_angle, param_dict_reg_v_a


@app.cell
def _(hssm_reg_v_a_angle):
    hssm_reg_v_a_angle
    return


@app.cell
def _(hssm_reg_v_a_angle, mo):
    hssm_reg_v_a_angle_graph = hssm_reg_v_a_angle.graph()

    hssm_reg_v_a_angle_png = hssm_reg_v_a_angle_graph.pipe(format="png")
    mo.image(hssm_reg_v_a_angle_png)
    return


@app.cell
def _(hssm_reg_v_a_angle):
    infer_data_reg_v_a = hssm_reg_v_a_angle.sample(
        sampler="mcmc",
        chains=2,
        cores=1,
        draws=1000,
        tune=1000,
        mp_ctx="spawn",
    )
    return (infer_data_reg_v_a,)


@app.cell
def _(az, infer_data_reg_v_a):
    az.summary(
        infer_data_reg_v_a, var_names=["~a", "~v"]
    )  # , var_names=["~rt,response_a"])
    return


@app.cell
def _(az, hssm_reg_v_a_angle, param_dict_reg_v_a, plt):
    az.plot_trace(
        hssm_reg_v_a_angle.traces,
        var_names=["~v", "~a"],
        lines=[(key_, {}, param_dict_reg_v_a[key_]) for key_ in param_dict_reg_v_a],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We successfully recover our regression betas for `a`! Moreover, no warning signs concerning our chains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 4: Categorical covariates
    """)
    return


@app.cell
def _(hssm, np):
    # Set up trial by trial parameters
    x_cat = np.random.choice(4, size=1000).astype(int)
    x_offset_cat = np.array([0, 1, -0.5, 0.75])

    y_cat = np.random.uniform(-1, 1, size=1000)
    v_y_cat = 0.3
    v_reg_v_cat = 0 + (v_y_cat * y_cat) + x_offset_cat[x_cat]

    # rest
    a_reg_v_cat = 1.5
    z_reg_v_cat = 0.5
    t_reg_v_cat = 0.1

    # base dataset
    dataset_reg_v_cat = hssm.simulate_data(
        model="ddm", theta=dict(v=v_reg_v_cat, a=a_reg_v_cat, z=z_reg_v_cat, t=t_reg_v_cat), size=1
    )

    # Adding covariates into the datsaframe
    dataset_reg_v_cat["x"] = x_cat
    dataset_reg_v_cat["y"] = y_cat
    return (dataset_reg_v_cat,)


@app.cell
def _(dataset_reg_v_cat, hssm):
    model_reg_v_cat = hssm.HSSM(
        data=dataset_reg_v_cat,
        model="angle",
        include=[
            {
                "name": "v",
                "formula": "v ~ 0 + C(x) + y",
                "link": "identity",
            }
        ],
    )
    return (model_reg_v_cat,)


@app.cell
def _(mo, model_reg_v_cat):
    model_reg_v_cat_graph = model_reg_v_cat.graph()

    model_reg_v_cat_png = model_reg_v_cat_graph.pipe(format="png")
    mo.image(model_reg_v_cat_png)
    return


@app.cell
def _(model_reg_v_cat):
    infer_data_reg_v_cat = model_reg_v_cat.sample(
        sampler="mcmc",
        chains=2,
        cores=1,
        draws=1000,
        tune=500,
        mp_ctx="spawn",
    )
    return (infer_data_reg_v_cat,)


@app.cell
def _(az, infer_data_reg_v_cat):
    az.plot_forest(infer_data_reg_v_cat, var_names=["~v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hierarchical Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try to fit a hierarchical model now. We will simulate a dataset with $15$ participants, with $200$ observations / trials for each participant.

    We define a group mean `mean_v` and a group standard deviation `sd_v` for the intercept parameter of the regression on `v`, which we sample from a corresponding normal distribution for each participant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate Data
    """)
    return


@app.cell
def _(hssm, np, pd):
    # Make some hierarchical data
    n_participants = 15  # number of participants
    n_trials = 200  # number of trials per participant

    sd_v = 0.5  # sd for v-intercept
    mean_v = 0.5  # mean for v-intercept

    data_list = []
    for i in range(n_participants):
        # Make parameters for participant i
        v_intercept_hier = np.random.normal(mean_v, sd_v, size=1)
        x_hier = np.random.uniform(-1, 1, size=n_trials)
        v_x_hier = 0.8
        y_hier = np.random.uniform(-1, 1, size=n_trials)
        v_y_hier = 0.3
        v_hier = v_intercept_hier + (v_x_hier * x_hier) + (v_y_hier * y_hier)

        a_hier = 1.5
        t_hier = 0.5
        z_hier = 0.5

        # true_values = np.column_stack(
        #     [v, np.repeat([[1.5, 0.5, 0.5, 0.0]], axis=0, repeats=n_trials)]
        # )

        data_tmp = hssm.simulate_data(
            model="ddm", theta=dict(v=v_hier, a=a_hier, z=z_hier, t=t_hier), size=1
        )
        data_tmp["participant_id"] = i
        data_tmp["x"] = x_hier
        data_tmp["y"] = y_hier

        data_list.append(data_tmp)

    # Make single dataframe out of participant-wise datasets
    dataset_reg_v_hier = pd.concat(data_list)
    dataset_reg_v_hier
    return (dataset_reg_v_hier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now define our `HSSM` model.

    We specify the regression as `v ~ 1 + (1|participant_id) + x + y`.

    `(1|participant_id)` tells the model to create a *participant-wise* offset for the intercept parameter. The rest of the regression $\beta$'s is fit globally.

    As an **R** user you may recognize this syntax from the [lmer](https://www.rdocumentation.org/packages/lme4/versions/1.1-33/topics/lmer) package.

    Our [Bambi](https://bambinos.github.io/bambi/) backend is essentially a Bayesian version of [lmer](https://www.rdocumentation.org/packages/lme4/versions/1.1-33/topics/lmer), quite like the [BRMS](https://cran.r-project.org/web/packages/brms/index.html) package in **R**, which operates on top of [STAN](https://mc-stan.org/).

    As a previous [HDDM](https://hddm.readthedocs.io/en/latest/) user, you may recognize that now proper mixed-effect models are viable!

    You should be able to handle between and within participant effects naturally now!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic Hierarchical Model
    """)
    return


@app.cell
def _(dataset_reg_v_hier, hssm):
    model_reg_v_angle_hier = hssm.HSSM(
        data=dataset_reg_v_hier,
        model="angle",
        noncentered=True,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {
                        "name": "Normal",
                        "mu": 0.0,
                        "sigma": 0.5,
                    },
                    "x": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                    "y": {"name": "Normal", "mu": 0.0, "sigma": 0.5},
                },
                "formula": "v ~ 1 + (1|participant_id) + x + y",
                "link": "identity",
            }
        ],
    )
    return (model_reg_v_angle_hier,)


@app.cell
def _(mo, model_reg_v_angle_hier):
    model_reg_v_angle_hier_graph = model_reg_v_angle_hier.graph()

    model_reg_v_angle_hier_graph_png = model_reg_v_angle_hier_graph.pipe(format="png")
    mo.image(model_reg_v_angle_hier_graph_png)
    return


@app.cell
def _(jax, model_reg_v_angle_hier):
    jax.config.update("jax_enable_x64", False)
    model_reg_v_angle_hier.sample(
        sampler="mcmc",
        chains=2,
        cores=2,
        draws=500,
        tune=500,
        mp_ctx="spawn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at the posteriors!
    """)
    return


@app.cell
def _(az, model_reg_v_angle_hier):
    az.plot_forest(model_reg_v_angle_hier.traces, var_names=["~v", "~a"], combined=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitting single models is all well and good. We are however, often interested in comparing how well a few different models account for the same data.

    Through [ArviZ](https://python.arviz.org/en/stable/index.html), we have out of the box access to modern Bayesian Model Comparison. We will keep it simple here, just to illustrate the basic idea.

    ### Scenario

    The following scenario is explored.

    First we generate data from a `ddm` model with fixed parameters, specifically we set the `a` parameter to $1.5$.

    We then define two `HSSM` models:

    1. A model which allows fitting all but the `a` parameter, which is fixed to $1.0$ (wrong)
    2. A model which allows fitting all but the `a` parameter, which is fixed to $1.5$ (correct)

    We then use the [ArviZ](https://python.arviz.org/en/stable/index.html)'s `compare()` function, to perform model comparison via `elpd_loo`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Simulation
    """)
    return


@app.cell
def _(hssm):
    # Parameters
    param_dict_mod_comp = dict(v=0.5, a=1.5, z=0.5, t=0.2)

    # Simulation
    dataset_model_comp = hssm.simulate_data(
        model="ddm", theta=param_dict_mod_comp, size=500
    )

    print(dataset_model_comp)
    return (dataset_model_comp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining the Models
    """)
    return


@app.cell
def _(dataset_model_comp, hssm):
    # 'wrong' model
    model_model_comp_1 = hssm.HSSM(
        data=dataset_model_comp,
        model="angle",
        a=1.0,
    )
    return (model_model_comp_1,)


@app.cell
def _(dataset_model_comp, hssm):
    # 'correct' model
    model_model_comp_2 = hssm.HSSM(
        data=dataset_model_comp,
        model="angle",
        a=1.5,
    )
    return (model_model_comp_2,)


@app.cell
def _(dataset_model_comp, hssm):
    # 'wrong' model ddm
    model_model_comp_3 = hssm.HSSM(
        data=dataset_model_comp,
        model="ddm",
        a=1.0,
    )
    return (model_model_comp_3,)


@app.cell
def _(model_model_comp_1):
    infer_data_model_comp_1 = model_model_comp_1.sample(
        sampler="mcmc",
        cores=1,
        chains=2,
        draws=1000,
        tune=1000,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
    )
    return


@app.cell
def _(model_model_comp_2):
    infer_data_model_comp_2 = model_model_comp_2.sample(
        sampler="mcmc",
        cores=1,
        chains=2,
        draws=1000,
        tune=1000,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
    )
    return


@app.cell
def _(model_model_comp_3):
    infer_data_model_comp_3 = model_model_comp_3.sample(
        sampler="mcmc",
        cores=1,
        chains=2,
        draws=1000,
        tune=1000,
        idata_kwargs=dict(
            log_likelihood=True
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compare
    """)
    return


@app.cell
def _(az, model_model_comp_1, model_model_comp_2, model_model_comp_3):
    compare_data = az.compare(
        {
            "a_fixed_1(wrong)": model_model_comp_1.traces,
            "a_fixed_1.5(correct)": model_model_comp_2.traces,
            "a_fixed_1_ddm(wrong)": model_model_comp_3.traces,
        }
    )

    compare_data
    return (compare_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the posterior weight on the `correct` model is close to (or equal to ) $1$ here.
    In other words model comparison points us to the correct model with
    a very high degree of certainty here!


    We can also use the `.plot_compare()` function to illustrate the model comparison visually.
    """)
    return


@app.cell
def _(az, compare_data):
    az.plot_compare(compare_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the forest plot we can take a look at what goes wrong for the "wrong" model.

    To make up for the mistplaced setting of the `a` parameter, the posterior seems to compensate by
    mis-estimating the other parameters.
    """)
    return


@app.cell
def _(az, model_model_comp_1, model_model_comp_2, model_model_comp_3):
    az.plot_forest(
        [model_model_comp_1.traces, model_model_comp_2.traces, model_model_comp_3.traces],
        model_names=["a_fixed_1(wrong)", "a_fixed_1.5(correct)", "a_fixed_1(wrong)_ddm"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Closer look!

    <center><img src="public/pytensor_jax.png"></center>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have seen a few examples of HSSM models at this point. Add a model via a string, maybe toy a bit with with the priors and set regression functions for a given parameter. Turn it hierarchical...  Here we begin to peak a bit under the hood.

    After all, we want to encourage you to contribute models to the package yourself.

    Let's remind ourself of the `model_config` dictionaries that define model properties for us. Again let's start with the DDM.
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"].keys()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The dictionary has a few high level keys.

    1. `response`

    2. `list_params`

    3. `description`

    4. `likelihoods`


    Let us take a look at the available `likelihoods`:
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the DDM we have available all three types of likelihoods that HSSM deals with:

    1. `analytical`
    2. `approx_differentiable`
    3. `blackbox`

    Let's expand the dictionary contents more:
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]["analytical"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see three properties (key) in this dictionary, of which two are essential:

    - The `loglik` field, which points to the likelihood function
    - The `backend` field, which can be either `None` (defaulting to pytensor for `analytical` likelihoods), `jax` or `pytensor`
    - The `bounds` field, which specifies bounds on a subset of the model parameters
    - The `default_priors` field, which specifies parameter wise priors

    If you provide `bounds` for a parameter, but no `default_priors`, a *Uniform* prior that respects the specified bounds will be applied.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's look at the `approx_differentiable` part.
    The likelihood in this part is based on a [LAN]() which was available in [HDDM]() through the [LAN extension]().
    """)
    return


@app.cell
def _(hssm):
    hssm.config.default_model_config["ddm"]["likelihoods"]["approx_differentiable"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that the `loglik` field is now a string that points to a `.onnx` file.
    [Onnx](https://onnx.ai/) is a meta framework for Neural Network specification, that allows translation between deep learning Frameworks. This is the preferred format for the neural networks we store in our model reservoir on [HuggingFace](https://huggingface.co/).

    Moreover notice that we now have a `backend` field. We allow for two primary backends in the `approx_differentiable` field.

    1. `pytensor`
    2. `jax`

    The `jax` backend assumes that your likelihood is described as a jax function, the `pytensor` backend assumes that your likelihood is described as a `pytensor` function. Ok not that surprising...

    We won't dwell on this here, however the key idea is to provide users with a large degree of flexibility in describing their likelihood functions and moreover to allow targeted optimization towards MCMC sampler types that [PyMC]() allows us to access.

    You can find a [dedicated tutorial](https://lnccbrown.github.io/HSSM/tutorial_likelihoods/#3-kinds-of-likelihoods) in the documentation, which describes the different likelihoods in much more detail.

    Instead, let's take a quick look at how these newfound insights can be used for custom model definition.
    """)
    return


@app.cell
def _(dataset_simulated, hssm):
    hssm_alternative_model = hssm.HSSM(
        data=dataset_simulated,
        model="ddm",
        loglik_kind="approx_differentiable",
    )
    return (hssm_alternative_model,)


@app.cell
def _(hssm_alternative_model):
    hssm_alternative_model.loglik_kind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case we actually built the model class with an `approx_differentiable` LAN likelihood, instead of the default `analytical` likelihood we used in the beginning of the tutorial. The assumed generative model remains the `ddm` however!
    """)
    return


@app.cell
def _(hssm_alternative_model):
    hssm_alternative_model.sample(
        sampler="mcmc",
        cores=1,
        chains=2,
        draws=1000,
        tune=1000,
        idata_kwargs=dict(
            log_likelihood=False
        ),  # model comparison metrics usually need this!
        mp_ctx="spawn",
    )
    return


@app.cell
def _(az, hssm_alternative_model):
    az.plot_forest(hssm_alternative_model.traces)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can take this further and specify a completely custom likelihood. See the [dedicated tutorial](https://lnccbrown.github.io/HSSM/tutorial_likelihoods/#using-custom-likelihoods) for more examples!

    We will see one specific example below to illustrate another type of likelihood function we have available for model building in HSSM, the *Blackbox* likelihood.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 'Blackbox' Likelihoods

    <center><img src="public/blackbox.png"></center>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is a **Blackbox Likelihood Function**?

    A *Blackbox Likelihood Function* is essentially any Python `callable` (function) that provides trial by trial likelihoods for your model of interest. What kind of computations are performed in this Python function is completely arbitrary.

    E.g. you could built a function that performs forward simulation from you model, constructs are kernel-density estimate for the resulting likelihood functions and evaluates your datapoints on this ad-hoc generated approximate likelihood.

    What I just described is a once state-of-the-art method of performing simulation based inference on Sequential Sampling models, a precursor to LANs if you will.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will do something simpler to keep it short and sweet, but really... the possibilities are endless!

    <br> </br>
    <center><img src="https://media.giphy.com/media/juq2OXONntuHfLc4Tn/giphy.gif"></center>
    <br> </br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulating simple dataset from the DDM
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As always, let's begin by generating some simple dataset.
    """)
    return


@app.cell
def _(hssm):
    # Set parameters
    param_dict_blackbox = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    # Simulate
    dataset_blackbox = hssm.simulate_data(model="ddm", theta=param_dict_blackbox, size=1000)
    return dataset_blackbox, param_dict_blackbox


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define the likelihood
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the fun part... we simply define a Python function `my_blackbox_loglik` which takes in our `data` as well as a bunch of model parameters (in our case the familiar `v`,`a`, `z`, `t` from the DDM).

    The function then does some arbitrary computation inside (in our case e.g. we pass the data and parameters to the DDM log-likelihood from our predecessor package HDDM).

    The important part is that inside `my_blackbox_loglik` anything can happen. We happen to call a little custom function that defines the likelihood of a DDM.

    **Fun fact:**
    It is de-facto the likelihood which is called by [HDDM](https://hddm.readthedocs.io/en/latest/).
    """)
    return


@app.cell
def _(hddm_wfpt, np):
    def my_blackbox_loglik(data, v, a, z, t, err=1e-8):
        """Create a custom blackbox likelihood function."""
        data = data[:, 0] * data[:, 1]
        data_nrows = data.shape[0]
        # Our function expects inputs as float64, but they are not guaranteed to
        # come in as such --> we type convert
        return hddm_wfpt.wfpt.wiener_logp_array(
            np.float64(data),
            (np.ones(data_nrows) * v).astype(np.float64),
            np.ones(data_nrows) * 0,
            (np.ones(data_nrows) * 2 * a).astype(np.float64),
            (np.ones(data_nrows) * z).astype(np.float64),
            np.ones(data_nrows) * 0,
            (np.ones(data_nrows) * t).astype(np.float64),
            np.ones(data_nrows) * 0,
            err,
            1,
        )
    return (my_blackbox_loglik,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define HSSM class with our Blackbox Likelihood
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now define our HSSM model class as usual, however passing our `my_blackbox_loglik()` function to the `loglik` argument, and passing as `loglik_kind = blackbox`.

    The rest of the model config is as usual. Here we can reuse our `ddm` model config, and simply specify bounds on the parameters (e.g. your Blackbox Likelihood might be trustworthy only on a restricted parameters space).
    """)
    return


@app.cell
def _(bmb, dataset_blackbox, hssm, my_blackbox_loglik):
    blackbox_model = hssm.HSSM(
        data=dataset_blackbox,
        model="ddm",
        loglik=my_blackbox_loglik,
        loglik_kind="blackbox",
        model_config={
            "bounds": {
                "v": (-10.0, 10.0),
                "a": (0.5, 5.0),
                "z": (0.0, 1.0),
            }
        },
        t=bmb.Prior("Uniform", lower=0.0, upper=2.0),
    )
    return (blackbox_model,)


@app.cell
def _(blackbox_model, mo):
    blackbox_model_graph = blackbox_model.graph()

    blackbox_model_graph_png = blackbox_model_graph.pipe(format="png")
    mo.image(blackbox_model_graph_png)
    return


@app.cell
def _(blackbox_model):
    sample = blackbox_model.sample()
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **NOTE**:

    Since *Blackbox likelihood functions* are assumed to not be differentiable, our default sampler for such likelihood functions is a `Slice` sampler. HSSM allows you to choose any other suitable sampler from the PyMC package instead. A bunch of options are available for gradient-free samplers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results
    """)
    return


@app.cell
def _(az, sample):
    az.summary(sample)
    return


@app.cell
def _(az, param_dict_blackbox, plt, sample):
    az.plot_trace(
        sample,
        lines=[(key_, {}, param_dict_blackbox[key_]) for key_ in param_dict_blackbox],
    )
    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## HSSM Random Variables in PyMC

    We covered a lot of ground in this tutorial so far. You are now a sophisticated HSSM user.

    It is therefore time to reveal a secret. We can actuallly peel back one more layer...
    <br> </br>
    <center><img src="https://media.giphy.com/media/3o6EhKIVvdhTkvjT8s/giphy.gif"></center>
    <br> </br>

    Instead of letting HSSM help you build the entire model, we can instead use HSSM to construct valid [PyMC](https://www.pymc.io/welcome.html) distributions and then proceed to build a custom PyMC model by ourselves...
    <br> </br>
    <center><img src="https://media.giphy.com/media/3pDxFfu5yZblPUVnWY/giphy.gif"></center>
    <br> </br>

    We will illustrate the simplest example below. It sets a pattern that can be exploited for much more complicated modeling exercises, which importantly go far beyond what our basic HSSM class may facilitate for you!

    See the [dedicated tutorial](https://lnccbrown.github.io/HSSM/notebooks/pymc/) in the [documentation](https://lnccbrown.github.io/HSSM/) if you are interested.

    Let's start by importing a few convenience functions:
    """)
    return


@app.cell
def _():
    # DDM models (the Wiener First-Passage Time distribution)
    from hssm.distribution_utils import make_distribution
    from hssm.likelihoods import DDM
    return DDM, make_distribution


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate some data
    """)
    return


@app.cell
def _(hssm):
    # Simulate
    param_dict_pymc = dict(v=0.5, a=1.5, z=0.5, t=0.5)

    dataset_pymc = hssm.simulate_data(model="ddm", theta=param_dict_pymc, size=1000)
    return dataset_pymc, param_dict_pymc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build a custom PyMC Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now use our custom random variable `DDM` directly in a PyMC model.
    """)
    return


@app.cell
def _(DDM, dataset_pymc):
    import pymc as pm

    with pm.Model() as ddm_pymc:
        v_custom = pm.Uniform("v", lower=-10.0, upper=10.0)
        a_custom = pm.HalfNormal("a", sigma=2.0)
        z_custom = pm.Uniform("z", lower=0.01, upper=0.99)
        t_custom = pm.Uniform("t", lower=0.0, upper=0.6)

        ddm = DDM(
            "DDM", observed=dataset_pymc[["rt", "response"]].values, v=v_custom, a=a_custom, z=z_custom, t=t_custom, 
        )
    return ddm_pymc, pm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the model graph:
    """)
    return


@app.cell
def _(ddm_pymc, mo, pm):
    ddm_pymc_graph = pm.model_to_graphviz(model=ddm_pymc)

    ddm_pymc_graph_png = ddm_pymc_graph.pipe(format="png")
    mo.image(ddm_pymc_graph_png)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looks remarkably close to our HSSM version!

    We can use PyMC directly to sample and finally return to ArviZ for some plotting!
    """)
    return


@app.cell
def _(ddm_pymc, pm):
    with ddm_pymc:
        ddm_pymc_trace = pm.sample()
    return (ddm_pymc_trace,)


@app.cell
def _(az, ddm_pymc_trace, param_dict_pymc, plt):
    az.plot_trace(
        ddm_pymc_trace,
        lines=[(key_, {}, param_dict_pymc[key_]) for key_ in param_dict_pymc],
    )

    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(az, ddm_pymc_trace):
    az.plot_forest(ddm_pymc_trace)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alternative Models with PyMC

    With very little extra work, we can in fact load any of the models accessible via HSSM. Here is an example, where we load the `angle` model instead.

    We first construction the likelihood function, using `make_likelihood_callable()`.

    Then we produce a valid `pymc.distribution` using the
    `make_distribution()` utility function.

    Just like the `DDM` class above, we can then use this distribution inside a **PyMC** model.
    """)
    return


@app.cell
def _(hssm, make_distribution):
    from hssm.distribution_utils import make_likelihood_callable

    angle_loglik_alt = make_likelihood_callable(
        loglik="angle.onnx",
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=[0, 0, 0, 0, 0],
    )

    ANGLE_alt = make_distribution(
        "angle",
        loglik=angle_loglik_alt,
        list_params=hssm.defaults.default_model_config["angle"]["list_params"],
    )
    return (make_likelihood_callable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that we need to supply the `params_is_reg` argument ("reg" for "regression").
    This is a boolean vector, which specifies for each input to the likelihood function, whether or not it is defined to be "trial-wise", as is expected if the parameter
    is the output e.g. of a regression function.
    """)
    return


@app.cell
def _(ANGLE, dataset_pymc, pm):
    # Angle pymc
    with pm.Model() as angle_pymc:
        # Define parameters
        v_alt = pm.Uniform("v", lower=-10.0, upper=10.0)
        a_alt = pm.Uniform("a", lower=0.5, upper=2.5)
        z_alt = pm.Uniform("z", lower=0.01, upper=0.99)
        t_alt = pm.Uniform("t", lower=0.0, upper=0.6)
        theta_alt = pm.Uniform("theta", lower=-0.1, upper=1.0)

        # Our RV
        angle_alt = ANGLE(
            "ANGLE",
            v=v_alt,
            a=a_alt,
            z=z_alt,
            t=t_alt,
            theta=theta_alt,
            observed=dataset_pymc[["rt", "response"]].values,
        )
    return (angle_pymc,)


@app.cell
def _(angle_pymc, pm):
    with angle_pymc:
        idata_object = pm.sample(nuts_sampler="numpyro")
    return (idata_object,)


@app.cell
def _(az, idata_object, param_dict_pymc, plt):
    az.plot_trace(
        idata_object, lines=[(key_, {}, param_dict_pymc[key_]) for key_ in param_dict_pymc]
    )

    plt.tight_layout()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regression via PyMC

    Finally to illustrate the usage of PyMC a little more elaborately, let us build a PyMC model with regression components.
    """)
    return


@app.cell
def _():
    from typing import Optional


    def make_params_is_reg_vec(
        reg_parameters: Optional[list] = None, parameter_names: Optional[list] = None
    ):
        """Make a list of Trues and Falses to indicate which parameters are vectors."""
        if (not isinstance(reg_parameters, list)) or (
            not isinstance(parameter_names, list)
        ):
            raise ValueError("Both reg_parameters and parameter_names should be lists")

        bool_list = [0] * len(parameter_names)
        for param in reg_parameters:
            bool_list[parameter_names.index(param)] = 1
        return bool_list
    return (make_params_is_reg_vec,)


@app.cell
def _(
    hssm,
    make_distribution,
    make_likelihood_callable,
    make_params_is_reg_vec,
    np,
    v_intercept,
    v_x,
    v_y,
    x,
    y,
):
    # Set up trial by trial parameters
    v_intercept_pymc_reg = 0.3
    x_pymc_reg = np.random.uniform(-1, 1, size=1000)
    v_x_pymc_reg = 0.8
    y_pymc_reg = np.random.uniform(-1, 1, size=1000)
    v_y_pymc_reg = 0.3
    v_pymc_reg = v_intercept + (v_x * x) + (v_y * y)

    param_dict_pymc_reg = dict(
        v_Intercept=v_intercept_pymc_reg,
        v_x=v_x_pymc_reg,
        v_y=v_y_pymc_reg,
        v=v_pymc_reg,
        a=1.5,
        z=0.5,
        t=0.1,
        theta=0.0,
    )

    # base dataset
    pymc_reg_data = hssm.simulate_data(model="ddm", theta=param_dict_pymc_reg, size=1)

    # Adding covariates into the datsaframe
    pymc_reg_data["x"] = x
    pymc_reg_data["y"] = y

    # Make the boolean vector for params_is_reg argument
    bool_param_reg = make_params_is_reg_vec(
        reg_parameters=["v"],
        parameter_names=hssm.defaults.default_model_config["angle"]["list_params"],
    )

    angle_loglik_reg = make_likelihood_callable(
        loglik="angle.onnx",
        loglik_kind="approx_differentiable",
        backend="jax",
        params_is_reg=bool_param_reg,
    )

    ANGLE_reg = make_distribution(
        "angle",
        loglik=angle_loglik_reg,
        list_params=hssm.defaults.default_model_config["angle"]["list_params"],
    )
    return (pymc_reg_data,)


@app.cell
def _(ANGLE, pm, pymc_reg_data):
    import pytensor.tensor as pt

    with pm.Model(
        coords={
            "idx": pymc_reg_data.index,
            "resp": ["rt", "response"],
            "features": ["x", "y"],
        }
    ) as pymc_model_reg:
        # Features
        x_ = pm.Data("x", pymc_reg_data["x"].values, dims="idx")
        y_ = pm.Data("y", pymc_reg_data["y"].values, dims="idx")
        # Target
        obs = pm.Data("obs", pymc_reg_data[["rt", "response"]].values, dims=("idx", "resp"))

        # Priors
        a_reg = pm.Uniform("a", lower=0.5, upper=2.5)
        z_reg = pm.Uniform("z", lower=0.01, upper=0.99)
        t_reg = pm.Uniform("t", lower=0.0, upper=0.6)
        theta_reg = pm.Uniform("theta", lower=-0.1, upper=1.0)
        v_Intercept = pm.Uniform("v_Intercept", lower=-3, upper=3)
        v_betas = pm.Normal("v_beta", mu=[0, 0], sigma=0.5, dims=("features"))

        # Regression equation
        v = pm.Deterministic(
            "v", v_Intercept + pt.stack([x_, y_], axis=1) @ v_betas, dims="idx"
        )

        # Our RV
        angle_reg = ANGLE(
            "angle",
            v=v.squeeze(),
            a=a_reg,
            z=z_reg,
            t=t_reg,
            theta=theta_reg,
            observed=obs,
            dims=("idx", "resp"),
        )
    return (pymc_model_reg,)


@app.cell
def _(pm, pymc_model_reg):
    with pymc_model_reg:
        idata_pymc_reg = pm.sample(
            nuts_sampler="numpyro", idata_kwargs={"log_likelihood": True}
        )
    return (idata_pymc_reg,)


@app.cell
def _(az, idata_pymc_reg):
    az.plot_forest(idata_pymc_reg, var_names=["~v"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All layers peeled back, the only limit in your modeling endeavors becomes the limit of the PyMC universe!

    <br> </br>

    <center><img src="https://media.giphy.com/media/4ydWTcMBjimLbT1CHi/giphy.gif"></center>


    <center> Enjoy the exploration! </center>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # End
    """)
    return


if __name__ == "__main__":
    app.run()
