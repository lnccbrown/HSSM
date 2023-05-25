# HSSM - Hierarchical Sequential Sampling Modeling 

## Class: HSSM

### Introduction

The Hierarchical Sequential Sampling Model (HSSM) class.

### Attributes

- `data`: A pandas DataFrame with at least two columns of "rt" and "response" indicating the response time and responses.
- `model`: The name of the model to use. Currently supported models are "ddm", "angle", "levy", "ornstein", "weibull", "race_no_bias_angle_4", "ddm_seq2_no_bias". If using a custom model, please pass "custom". Defaults to "ddm".
- `include`: A list of dictionaries specifying parameter specifications to include in the model. If left unspecified, defaults will be used for all parameter specifications. Defaults to None.
- `list_params`: The list of strs of parameter names.
- `model_name`: The name of the model.
- `model_config`: A dictionary representing the model configuration.
- `model_distribution`: The likelihood function of the model in the form of a pm.Distribution subclass.
- `family`: A Bambi family object.
- `priors`: A dictionary containing the prior distribution of parameters.
- `formula`: A string representing the model formula.
- `link`: A string or a dictionary representing the link functions for all parameters.
- `params`: A list of Param objects representing model parameters.

### Methods

- `__init__`: Constructor method that initializes the HSSM class.
- `sample`: A method to sample posterior distributions.
- `set_alias`: Sets the alias for a parameter.
- `graph`: Plot the model with PyMC's built-in graph function.

## Method: __init__

### Description

Constructor method that initializes the HSSM class.

### Parameters

- `data`: A pandas DataFrame.
- `model`: SupportedModels, defaults to "ddm".
- `include`: A list of dictionaries or None, defaults to None.
- `model_config`: Config or None, defaults to None.
- `loglik`: LogLikeFunc or pytensor.graph.Op or None, defaults to None.
- `**kwargs`: Other parameters.

## Method: sample

TO-DO: Add Description and Parameters for this method.

## Method: set_alias

TO-DO: Add Description and Parameters for this method.

## Method: graph

TO-DO: Add Description and Parameters for this method.

::: hssm.hssm