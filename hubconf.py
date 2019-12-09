dependencies = ['torch']
from feature_loader import getEncoder, getAR, loadArgs
from cpc_default_config import get_default_cpc_config
import argparse
from model import CPCModel as cpcmodel
from model import CPCBertModel as cpcbert

def CPCModel(pretrained=False, *args, **kwargs):
    """
    Progressive growing model
    pretrained (bool): load a pretrained model ?
    encoder_type (string):
        -- "CPC" (default), "MFCC" or "LFB"
    arMode (string):
        -- "GRU" (default), "LSTM", "RNN"
    hiddenEncoder (int): size of hidden encoder layer
    normMode (string): defines the normalization layer in CPC
        -- "batchNorm" (default), "instanceNorm", "ID" or "layerNorm"
    sizeWindow (int): Expected size of the input sequence of the
        AR transformer *160
    abspos (bool)
    samplingType (string):
        -- "sequential" or "samespeaker"
    hiddenGar (int): dimension of encoding GRU layer
    nLevelsGRU (int): number of layers of the CPCAR network
    cpc_mode (string):
        -- "normal" (default), "bert", "reverse"
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
    if locArgs.cpc_mode == "bert":
        model = cpcbert(encoderNet, arNet, blockSize=locArgs.nPredicts)
        model.supervised = locArgs.supervised
    else:
        model = cpcmodel(encoderNet, arNet)
    if pretrained:
        model.load_state_dict(checkpoint["weights"], strict=False)
    return model
