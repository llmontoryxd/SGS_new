import numpy as np
from scipy.spatial.distance import pdist, squareform
from math import sqrt
from tqdm import tqdm

from models import Params
from varcovar import varcovar


def frob(params: Params, CY, debug=False, nnc=False):
    err = np.zeros(params.m)
    X = np.linspace(1, params.nx, num=params.nx)
    Y = np.linspace(1, params.ny, num=params.ny)
    X, Y = np.meshgrid(X, Y)

    Yf = np.transpose(Y).flatten()
    Xf = np.transpose(X).flatten()
    XY = np.zeros((len(Xf), 2))
    for i in range(np.shape(XY)[0]):
        XY[i] = [Xf[i], Yf[i]]

    DIST = squareform(pdist(np.matmul(XY, params.covar.cx)))
    CY_true = np.kron(params.covar.g(DIST), params.covar.c0)

    for m in tqdm(range(len(err))):
        err[m] = sqrt(np.sum((CY[:, :, m]-CY_true)**2)) / sqrt(np.sum(CY_true))

    return err
