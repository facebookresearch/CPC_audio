import matplotlib.pyplot as plt
import numpy as np

def plot_hist(ax, vals, title, nbins, weight = None):
    hist, bin_edges = np.histogram(vals, bins=nbins)
    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    divisor = len(vals) if weight is None else np.sum(weight)
    ax.bar(center, hist / divisor, align='center', width=width, label=title)


def plot_edge_hist(ax, vals, title, nbins, weight = None):
    hist, bin_edges = np.histogram(vals, bins=nbins, weights=weight)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    divisor = len(vals) if weight is None else np.sum(weight)
    ax.plot(center, hist / divisor, label=title)

def plot_as_hist(ax, vals, title):
    center = np.arange(len(vals))
    ax.bar(center, np.sort(vals), label=title)
