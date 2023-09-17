import matplotlib.pyplot as plt
from math import sqrt, floor

import numpy as np
import time

from histo import histo, qq_plot
from vario import vario


def plot(Rest, means, std, grid, covar, m, save=False, show=False, show_NNC=False, mode=None):
    if not show_NNC:
        if mode is None:
            cols = floor(sqrt(m))
            rows = cols
            while m > cols*rows:
                rows += 1
        else:
            cols = 2
            rows = m
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 9*m), squeeze=False)

        if mode is None:
            m_now = 0
            for i in range(rows):
                for j in range(cols):
                    if m_now >= m:
                        break
                    c = axs[i, j].imshow(Rest[:, :, m_now])
                    plt.colorbar(c, ax=axs[i, j])
                    axs[i, j].set(xlabel='x', ylabel='y', title=f'mean = {means[i+j]}')
                    m_now += 1
        else:
            for i in range(rows):
                c = axs[i, 0].imshow(Rest[:, :, i])
                plt.colorbar(c, ax=axs[i, 0])
                axs[i, 0].set(xlabel='x', ylabel='y', title=f'mean = {means[i]}')

                match mode:
                    case 'histo':
                        bins, fit = histo(Rest[:, :, i], means[i], std[i])
                        axs[i, 1].hist(bins, color='blue', bins=10, density=True, rwidth=2)

                        axs[i, 1].plot(bins, fit, linestyle='-', color='black')
                        axs[i, 1].set(xlabel='Values', ylabel='Density')
                        axs[i, 1].grid()
                    case 'qq':
                        sorted_random_dist, norm_one_rest, lin_reg_func = qq_plot(Rest[:, :, i], means[i], std[i])
                        axs[i, 1].plot(sorted_random_dist, norm_one_rest, color='red', linestyle='None', marker='o', markersize=2)
                        axs[i, 1].plot(sorted_random_dist, lin_reg_func, color='black', linestyle='--')
                        axs[i, 1].set(xlabel='Theoretical quantities', ylabel='Normalized values')
                    case 'vario':
                        bin_centers, bin_means, lag, initial_var = vario(grid, Rest[:, :, i], covar)
                        axs[i, 1].plot(bin_centers, bin_means, linestyle='None', marker='o', color='red')
                        axs[i, 1].plot(lag, initial_var, linestyle='None', marker='o', color='black', label='fit')
                        axs[i, 1].grid()


    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        one_rest = Rest[:, :, 0]
        data = np.concatenate((one_rest, one_rest), axis=0)
        data = np.concatenate((data, data), axis=1)
        x_c = np.shape(data)[1]/2
        y_c = np.shape(data)[0]/2
        c = ax.imshow(data)
        ax.plot(np.zeros(np.shape(data)[0])+x_c, np.linspace(0, np.shape(data)[0]-1, num=np.shape(data)[0]),
                color='black')
        ax.plot(np.linspace(0, np.shape(data)[1] - 1, num=np.shape(data)[1]),
                np.zeros(np.shape(data)[1])+y_c,
                color='black')
        plt.colorbar(c, ax=ax)
        ax.set(xlabel='x', ylabel='y', title=f'mean = {means[0]}')

    if show:
        plt.show()
    if save:
        fig.savefig('outdata/'+time.strftime('%y-%m-%d-%H-%M-%S'))

