# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "cpc.eval.ABX.dtw",
        ["cpc/eval/ABX/dtw.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name='CPC_audio',
    version='1.0',
    description='An implementation of the contrast predictive coding (CPC) '
    'training method for audio data.',
    author='Facebook AI Research',
    packages=find_packages(),
    classifiers=["License :: OSI Approved :: MIT License",
                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering",
                 "Programming Language :: Python"],
    ext_modules=cythonize(extensions, language_level="3")
)
