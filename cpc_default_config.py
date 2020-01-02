import argparse


def get_default_cpc_config():
    parser = set_default_cpc_config(argparse.ArgumentParser())
    return parser.parse_args([])


def set_default_cpc_config(parser):
    # Run parameters

    parser.add_argument('--hiddenEncoder', type=int, default=256)
    parser.add_argument('--hiddenGar', type=int, default=256)
    parser.add_argument('--nPredicts', type=int, default=12)
    parser.add_argument('--negativeSamplingExt', type=int, default=128)
    parser.add_argument('--learningRate', type=float, default=2e-4)
    parser.add_argument('--schedulerStep', type=int, default=-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-08)
    parser.add_argument('--sizeWindow', type=int, default=20480)
    parser.add_argument('--nEpoch', type=int, default=200)
    parser.add_argument('--samplingType', type=str, default='samespeaker',
                        choices=['samespeaker', 'uniform',
                                 'samesequence', 'sequential'])
    parser.add_argument('--nLevelsGRU', type=int, default=1)
    parser.add_argument('--nLevelsPhone', type=int, default=1)
    parser.add_argument('--abspos', action='store_true')
    parser.add_argument('--cpc_mode', type=str, default=None,
                        choices=['reverse', 'bert', 'none'])
    parser.add_argument('--encoder_type', type=str,
                        choices=['cpc', 'mfcc', 'lfb'],
                        default='cpc')
    parser.add_argument('--onEncoder', action='store_true')
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--speakerEmbedding', type=int, default=0)
    parser.add_argument('--arMode', default='LSTM',
                        choices=['GRU', 'LSTM', 'RNN', 'no_ar', 'transformer'])
    parser.add_argument('--normMode', type=str, default='layerNorm',
                        choices=['instanceNorm', 'ID', 'layerNorm',
                                 'batchNorm'])
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--rnnMode', type=str, default='transformer',
                        choices=['transformer', 'RNN', 'LSTM', 'linear',
                                 'ffd', 'conv4', 'conv8', 'conv12'])
    parser.add_argument('--clustering', type=str, default=None,
                        choices=['deepEmbedded', 'deepClustering',
                                 'CTCClustering'])
    parser.add_argument('--n_clusters', type=int, default=200)
    parser.add_argument('--cluster_delay', type=int, default=0)
    parser.add_argument('--cluster_iter', type=int, default=100)
    parser.add_argument('--clustering_update', type=str, default='kmean',
                        choices=['kmean', 'dpmean'])
    parser.add_argument('--schedulerRamp', type=int, default=None)

    return parser
