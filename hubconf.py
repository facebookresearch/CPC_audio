# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
from cpc.model import CPCModel as cpcmodel
from cpc.cpc_default_config import get_default_cpc_config
from cpc.feature_loader import getEncoder, getAR, loadArgs
dependencies = ['torch', 'torchaudio']


def CPC_audio(pretrained=False,
              **kwargs):
    """
    Contrast predictive learning model for audio data
    pretrained: if True, load a model trained on libri-light 60k
    (https://arxiv.org/abs/1912.07875)
    **kwargs : see cpc/cpc_default_config to get the list of possible arguments
    """
    locArgs = get_default_cpc_config()
    if pretrained:
        checkpoint_url = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url,
                                                        progress=False)
        loadArgs(locArgs, argparse.Namespace(**checkpoint["config"]))
    else:
        args = argparse.Namespace(**kwargs)
        loadArgs(locArgs, args)
    encoderNet = getEncoder(locArgs)
    arNet = getAR(locArgs)
    model = cpcmodel(encoderNet, arNet)
    if pretrained:
        model.load_state_dict(checkpoint["weights"], strict=False)
    return model
