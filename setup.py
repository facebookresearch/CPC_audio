from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "cpc.eval.ABX.dtw",
        ["cpc/eval/ABX/dtw.pyx"],
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
    ext_modules = cythonize(extensions, language_level = "3")
)
