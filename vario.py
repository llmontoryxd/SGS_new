import scipy.spatial.distance as distance
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from typing import Callable
from numpy.typing import ArrayLike

import model


def semi_variance(Rest):
    vals = Rest.flatten()
    n = len(vals)
    semi_vmatrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            semi_vmatrix[i, j] = 0.5*(vals[i] - vals[j])**2
    return semi_vmatrix


def cloud(grid, Rest, num_show=1000):
    lag = distance.squareform(distance.pdist(grid))
    semi_vmatrix = semi_variance(Rest)
    show_idxs = random.sample(range(len(lag)), num_show)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    ax.plot(lag.flatten()[show_idxs], semi_vmatrix.flatten()[show_idxs], linestyle='None', marker='o', color='black')
    ax.set(xlabel='Lag distance', ylabel='Semi-variance')
    plt.show()

    return lag.flatten()[show_idxs], semi_vmatrix.flatten()[show_idxs]


def vario(grid, Rest, covar, cut_off=100, bins=20):
    lag = distance.squareform(distance.pdist(grid))
    lag = lag.flatten()
    semi_vmatrix = semi_variance(Rest)
    semi_vmatrix = semi_vmatrix.flatten()
    mask = np.where(lag <= cut_off)
    lag = lag[mask]
    semi_vmatrix = semi_vmatrix[mask]

    statistic, bin_edges, bin_number = stats.binned_statistic(lag, semi_vmatrix, statistic='mean', bins=bins, range=None)
    bin_means = statistic
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    match covar.model:
        case 'gaussian':
            var: Callable[[ArrayLike], ArrayLike] = lambda h: covar.c0*(1 - np.exp(-h**2/covar.range[0]**2))
            #var: Callable[[ArrayLike], ArrayLike] = lambda h: model.gaussian(h, covar.range0[0], covar.c0)
        case 'exponential':
            var: Callable[[ArrayLike], ArrayLike] = lambda h: covar.c0*(1 - np.exp(-h/covar.range[0]))
            #var: Callable[[ArrayLike], ArrayLike] = lambda h: model.exponential(h, covar.range0[0], covar.c0)
        case 'spherical':
            var: Callable[[ArrayLike], ArrayLike] = lambda h: covar.c0*(3/2*np.minimum(h/covar.range[0], 1)-1/2*np.minimum(h/covar.range[0], 1)**3)
            #var: Callable[[ArrayLike], ArrayLike] = lambda h: model.spherical(h, covar.range0[0], covar.c0)
        case _:
            raise ValueError('Your model is not supported yet')

    initial_var = var(lag)

    return bin_centers, bin_means, lag, initial_var


