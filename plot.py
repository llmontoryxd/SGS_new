import matplotlib.pyplot as plt
from math import sqrt, floor

import numpy as np
import time

from tqdm import tqdm

from histo import histo, qq_plot, histo_U
from vario import vario
from models import Plot


def plot(Rest, means, std, grid, covar, m, err: None|list, U, plot_params: Plot):
    if not plot_params.show_NNC:
        if plot_params.mode is None:
            cols = floor(sqrt(m))
            rows = cols
            while m > cols*rows:
                rows += 1
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 9 * m), squeeze=False)
        elif plot_params.mode == 'all':
            cols = 4
            rows = m
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16*2, 9 * m), squeeze=False)
        else:
            cols = 2
            rows = m
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 9 * m), squeeze=False)

        if plot_params.mode is None:
            m_now = 0
            for i in tqdm(range(rows)):
                for j in range(cols):
                    if m_now >= m:
                        break
                    c = axs[i, j].imshow(Rest[:, :, m_now])
                    plt.colorbar(c, ax=axs[i, j])
                    title_str = f'mean = {means[i+j]}'
                    if err is not None:
                        title_str += f', err = {err[i+j]}'
                    axs[i, j].set(xlabel='x', ylabel='y', title=title_str)
                    m_now += 1
        else:
            for i in tqdm(range(rows)):
                c = axs[i, 0].imshow(Rest[:, :, i])
                plt.colorbar(c, ax=axs[i, 0])
                title_str = f'mean = {means[i]}'
                if err is not None:
                    title_str += f', err = {err[i]}'
                axs[i, 0].set(xlabel='x', ylabel='y', title=title_str)

                match plot_params.mode:
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
                        axs[i, 1].set(xlabel='Theoretical quantiles', ylabel='Sample quantiles')
                    case 'vario':
                        bin_centers, bin_means, lag, initial_var = vario(grid, Rest[:, :, i], covar,
                                                                         plot_params.cutoff, plot_params.bins)
                        axs[i, 1].plot(bin_centers, bin_means, linestyle='None', marker='o', color='red')
                        axs[i, 1].plot(lag, initial_var, linestyle='None', marker='o', color='black', label='fit')
                        axs[i, 1].set(ylabel=r'$\gamma$', xlabel='lag distance')
                        axs[i, 1].grid()
                    case 'check_U':
                        bins, fit = histo_U(U)
                        axs[i, 1].hist(bins, color='blue', bins=10, density=True, rwidth=2)

                        axs[i, 1].plot(bins, fit, linestyle='-', color='black')
                        axs[i, 1].set(xlabel='Values', ylabel='Density')
                        axs[i, 1].grid()
                    case 'all':
                        bins, fit = histo(Rest[:, :, i], means[i], std[i])
                        axs[i, 1].hist(bins, color='blue', bins=10, density=True, rwidth=2)

                        axs[i, 1].plot(bins, fit, linestyle='-', color='black')
                        axs[i, 1].set(xlabel='Values', ylabel='Density')
                        axs[i, 1].grid()

                        sorted_random_dist, norm_one_rest, lin_reg_func = qq_plot(Rest[:, :, i], means[i], std[i])
                        axs[i, 2].plot(sorted_random_dist, norm_one_rest, color='red', linestyle='None', marker='o',
                               markersize=2)
                        axs[i, 2].plot(sorted_random_dist, lin_reg_func, color='black', linestyle='--')
                        axs[i, 2].set(xlabel='Theoretical quantiles', ylabel='Sample quantiles')

                        bin_centers, bin_means, lag, initial_var = vario(grid, Rest[:, :, i], covar,
                                                                         plot_params.cutoff, plot_params.bins)
                        axs[i, 3].plot(bin_centers, bin_means, linestyle='None', marker='o', color='red')
                        axs[i, 3].plot(lag, initial_var, linestyle='None', marker='o', color='black', label='fit')
                        axs[i, 3].set(ylabel=r'$\gamma$', xlabel='lag distance')
                        axs[i, 3].grid()


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

    if plot_params.show:
        plt.show()
    if plot_params.save:
        if plot_params.savefilename == 'gen':
            savefilename = time.strftime('%y-%m-%d-%H-%M-%S')
        else:
            savefilename = plot_params.savefilename
        fig.savefig('outdata/'+savefilename)

