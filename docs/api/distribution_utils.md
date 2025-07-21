The `hssm.distribution_utils` contains useful functions for building `pm.Distribution`
classes. Other than the `download_hf` function that downloads ONNX models shared on our
[huggingface model repository](https://huggingface.co/franklab/HSSM/tree/main), you will
generally not have to use these functions. For advanced users who want to build their
own PyMC models, they can use these functions to create `pm.Distribution` and
`RandomVariable` classes that they desire.

::: hssm.distribution_utils
    handler: python
    options:
        members:
        - download_hf
        - load_onnx_model
        - make_distribution
        - make_ssm_rv
        - make_family
        - make_likelihood_callable
        - make_missing_data_callable
        - make_blackbox_op
        - assemble_callables
