# noqa: D100
import platform

import numpy as np  # noqa
from setuptools import Extension, setup  # noqa

try:
    from Cython.Build import cythonize

    if platform.system() == "Darwin":
        ext1 = Extension(
            "wfpt",
            ["src/hssm/likelihoods/hddm_wfpt/wfpt.pyx"],
            language="c++",
            extra_compile_args=["-stdlib=libc++"],
            extra_link_args=["-stdlib=libc++", "-mmacosx-version-min=10.9"],
        )
    else:
        ext1 = Extension(
            "wfpt", ["src/hssm/likelihoods/hddm_wfpt/wfpt.pyx"], language="c++"
        )

    ext_modules = cythonize(
        [
            ext1,
            Extension(
                "cdfdif_wrapper",
                [
                    "src/hssm/likelihoods/hddm_wfpt/cdfdif_wrapper.pyx",
                    "src/hssm/likelihoods/hddm_wfpt/cdfdif.c",
                ],
            ),
        ],
        compiler_directives={"language_level": "3", "linetrace": True},
    )

except ImportError:
    ext_modules = [
        Extension("wfpt", ["src/hssm/likelihoods/hddm_wfpt/wfpt.cpp"], language="c++"),
        Extension(
            "cdfdif_wrapper",
            [
                "src/hssm/likelihoods/hddm_wfpt/cdfdif_wrapper.c",
                "src/hssm/likelihoods/hddm_wfpt/cdfdif.c",
            ],
        ),
    ]
