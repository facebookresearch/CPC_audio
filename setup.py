from setuptools import setup
from setuptools import find_packages
from pathlib import Path

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
)
