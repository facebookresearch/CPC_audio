The installation is a tiny bit involved due to the torch-audio dependency.

0/ Clone the repo:
`git clone git@github.com:fairinternal/CPC_torch.git && cd CPC_torch`

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f setup/environment.yml && conda activate cpc37`

3/ Go to the root directory and run setup.py
`pip install -e .`

# CUDA driver

This setup is given for CUDA 9.2 if you use a different version of CUDA then please change the version of cudatoolkit in environment.yml.
For more information on the cudatoolkit version to use, please check https://pytorch.org/

# (FAIR only) Slurm-based grid search

This feature requires submitit, which can be installed by running:
`pip install git+ssh://git@github.com/fairinternal/submitit@master#egg=submitit`
