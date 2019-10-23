import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np


def plotHist(seq, nBins, pathOut, y_label="", normalized=True):

    if isinstance(seq, list):
        seq = np.array(seq)

    counts, bins = np.histogram(seq, bins=nBins)
    if normalized:
        counts = counts / np.sum(counts)
    plt.clf()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(pathOut)


def plotScatter(seqs, xLabel, pathOut, x_label="", y_label="", title=""):
    plt.clf()
    for i in range(seqs.shape[0]):
        plt.scatter(xLabel, seqs[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pathOut)


def plotSeq(seqs, xLabel, pathOut, x_label="", y_label="", title="",
            xscale="linear", yscale="linear", legend=None):
    plt.clf()
    for i in range(seqs.shape[0]):
        plt.plot(xLabel, seqs[i])
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend is not None:
        plt.gca().legend(legend)
    plt.tight_layout()
    plt.savefig(pathOut)
