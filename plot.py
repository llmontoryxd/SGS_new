import matplotlib.pyplot as plt
from math import sqrt, floor

import numpy as np
import time


def plot(Rest, m, save=False, show=False, show_NNC=False):
    if not show_NNC:
        cols = floor(sqrt(m))
        rows = cols
        if m > cols**2:
            rows += 1

        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8), squeeze=False)
        m_now = 0

        for i in range(rows):
            for j in range(cols):
                if m_now >= m:
                    break
                c = axs[i, j].imshow(Rest[:, :, m_now])
                plt.colorbar(c, ax=axs[i, j])
                axs[i, j].set(xlabel='x', ylabel='y')
                m_now += 1
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        one_rest = Rest[:, :, 0]
        data = np.concatenate((one_rest, one_rest), axis=0)
        data = np.concatenate((data, data), axis=1)
        c = ax.imshow(data)
        plt.colorbar(c, ax=ax)
        ax.set(xlabel='x', ylabel='y')


    if show:
        plt.show()
    if save:
        fig.savefig('outdata/'+time.strftime('%y-%m-%d-%H-%M-%S'))

