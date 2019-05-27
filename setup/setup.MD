The installation is a tiny bit involved due to the torch-audio dependency.

0/ Clone the repo:
`git clone git@github.com:fairinternal/CPC_torch.git && cd CPC_torch`

1/ Install libraries which would be required for torch-audio https://github.com/pytorch/audio :
 * MacOS: `brew install sox`
 * Linux: `sudo apt-get install sox libsox-dev libsox-fmt-all`

2/ `conda env create -f setup/environment.yml && conda activate cpc37`

3/ install torch-audio:
  * Linux: `pip install git+https://github.com/pytorch/audio.git`
  * MacOS: check the instruction on the site https://github.com/pytorch/audio
