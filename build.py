# noqa: D100
import os
import platform
import shutil
from distutils.core import Distribution, Extension

import numpy as np  # noqa
from Cython.Build import build_ext, cythonize

cython_dir = "src/hssm/likelihoods/hddm_wfpt"

try:
    if platform.system() == "Darwin":
        ext1 = Extension(
            "wfpt",
            ["src/hssm/likelihoods/hddm_wfpt/wfpt.pyx"],
            language="c++",
            extra_compile_args=["-stdlib=libc++"],
            include_dirs=[np.get_include()],
            extra_link_args=["-stdlib=libc++", "-mmacosx-version-min=10.9"],
        )
    else:
        ext1 = Extension(
            "wfpt",
            ["src/hssm/likelihoods/hddm_wfpt/wfpt.pyx"],
            language="c++",
            include_dirs=[np.get_include()],
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
                include_dirs=[np.get_include()],
            ),
        ],
        include_path=[cython_dir],
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

dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)
