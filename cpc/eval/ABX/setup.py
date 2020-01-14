# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("dtw.pyx")
)
