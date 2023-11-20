import numpy as np
import random
from math import ceil, sqrt
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import inv
from tqdm import tqdm

import model
from models import Params
import k3d


def create_grid(params: Params):
    X = np.linspace(1, params.nx, num=params.nx)
    if params.ndim == 1:
        return X, X
    if params.ndim == 2:
        Y = np.linspace(1, params.ny, num=params.ny)
        grid = np.zeros((len(X), len(Y), 2))
        for k in range(np.shape(grid)[2]):
            for j in range(np.shape(grid)[1]):
                for i in range(np.shape(grid)[0]):
                    grid[i, j, :] = [X[i], Y[j]]

        grid = grid.reshape(np.shape(grid)[0] * np.shape(grid)[1], 2)
        X, Y = np.meshgrid(X, Y)

        return X, Y, grid
    if params.ndim == 3:
        Y = np.linspace(1, params.ny, num=params.ny)
        Z = np.linspace(1, params.nz, num=params.nz)
        grid = np.zeros((len(X), len(Y), len(Z), 3))
        for l in range(np.shape(grid)[3]):
            for k in range(np.shape(grid)[2]):
                for j in range(np.shape(grid)[1]):
                    for i in range(np.shape(grid)[0]):
                        grid[i, j, k, :] = [X[i], Y[j], Z[k]]
        grid = grid.reshape(np.shape(grid)[0] * np.shape(grid)[1] * np.shape(grid)[2], 3)
        X, Y, Z = np.meshgrid(X, Y, Z)

        return X, Y, Z, grid


def trad_sgs(params, data):
    '''
    Input:  (data)   <N,3> NumPy array of data
            (hs)     NumPy array of distances
            (bw)     bandwidth of the semivariogram
            (xs)     number of cells in the x dimension
            (ys)     number of cells in the y dimension
    Output: (M)      <xsteps,ysteps> NumPy array of data
                     representing the simulated distribution
                     of the variable of interest
    '''
    covfct = model.covariance(model.spherical, (params.covar.range0[0], params.covar.c0))

    # creating grid
    if params.ndim == 1:
        X, grid = create_grid(params)
    elif params.ndim == 2:
        X, Y, grid = create_grid(params)
    elif params.ndim == 3:
        X, Y, Z, grid = create_grid(params)
        # print(X, Y, Z)
    else:
        raise ValueError('Only works in 1, 2 or 3D')

    if params.ndim == 1:
        Rest = np.zeros((params.nx, params.m)) + np.nan
    elif params.ndim == 2:
        Rest = np.zeros((params.nx, params.ny, params.m)) + np.nan
    elif params.ndim == 3:
        Rest = np.zeros((params.nx, params.ny, params.nz, params.m)) + np.nan
    else:
        raise ValueError('Only works in 1, 2 or 3D')

    Rest_means = np.zeros(params.m)
    Rest_std = np.zeros(params.m)
    LambdaM = np.zeros((params.nx * params.ny, params.nx * params.ny, params.m))
    CY = np.zeros((params.nx * params.ny, params.nx * params.ny, params.m))

    for i_real in tqdm(range(params.m)):
        # Path generation
        if params.ndim == 1:
            Res = np.zeros((params.nx)) + np.nan
            Path = np.zeros((params.nx)) + np.nan
        elif params.ndim == 2:
            Res = np.zeros((params.nx, params.ny)) + np.nan
            Path = np.zeros((params.nx, params.ny)) + np.nan
        elif params.ndim == 3:
            Res = np.zeros((params.nx, params.ny, params.nz)) + np.nan
            Path = np.zeros((params.nx, params.ny, params.nz)) + np.nan
        else:
            raise ValueError('Only works in 1, 2 or 3D')

        Path = Path.flatten()
        np.random.seed(params.seed_path)
        id = np.argwhere(np.isnan(Path.flatten())).T[0]
        path = id[np.random.permutation(len(id))]

        Path[path] = range(len(id))
        Xf = np.transpose(X).flatten()
        if params.ndim == 1:
            Path = Path.reshape((params.nx))
            XY_i = Xf
        elif params.ndim == 2:
            Path = Path.reshape((params.nx, params.ny))
            Yf = np.transpose(Y).flatten()
            XY_i = np.zeros((len(Xf), 2))
            for i in range(np.shape(XY_i)[0]):
                XY_i[i] = [Xf[path[i]], Yf[path[i]]]
        elif params.ndim == 3:
            Path = Path.reshape((params.nx, params.ny, params.nz))
            Yf = np.transpose(Y).flatten()
            Zf = np.transpose(Z).flatten()
            XY_i = np.zeros((len(Xf), 3))
            for i in range(np.shape(XY_i)[0]):
                XY_i[i] = [Xf[path[i]], Yf[path[i]], Zf[path[i]]]
        else:
            raise ValueError('Only works in 1, 2 or 3D')

        # Get U from normal distribution
        np.random.seed(params.seed_U)
        if params.ndim == 1:
            U = np.random.randn(params.nx).flatten()
        elif params.ndim == 2:
            U = np.random.randn(params.nx, params.ny).flatten()
        elif params.ndim == 3:
            U = np.random.randn(params.nx, params.ny, params.nz).flatten()
        else:
            raise ValueError('Only works in 1, 2 or 3D')

        for i_pt in tqdm(range(np.shape(XY_i)[0])):
            #print(XY_i[i_pt])
            if type(data) is list and not data:
                Res = Res.flatten()
                Res[path[i_pt]] = U[i_pt] * np.sum([params.covar.c0])
                data = [[int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), Res[path[i_pt]]]]
            else:
                est, kstd = k3d.simple(params, data, covfct, XY_i[i_pt].astype(int), max_dist=7)
                Res = Res.flatten()
                if est is None:
                    Res[path[i_pt]] = U[i_pt] * np.sum([params.covar.c0])
                else:
                    Res[path[i_pt]] = est + U[i_pt] * kstd

                newdata = [int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), Res[path[i_pt]]]
                data = np.vstack((data, newdata))

        if params.ndim == 1:
            Res = Res.reshape((params.nx))
        if params.ndim == 2:
            Res = Res.reshape((params.nx, params.ny))
        if params.ndim == 3:
            Res = Res.reshape((params.nx, params.ny, params.nz))

        Rest[:, :, i_real] = Res


    for m in range(np.shape(Rest)[2]):
        Rest_means[m] = np.mean(Rest[:, :, m])
        Rest_std[m] = np.std(Rest[:, :, m])
    Rest = Rest + params.mean
    Rest_means = Rest_means + params.mean

    return Rest, Rest_means, Rest_std, grid, CY, U





    # # create array for the output
    # M = np.zeros((xs,ys))
    # # for each cell in the grid..
    # for step in path :
    #     # grab the index, the cell address, and the physical location
    #     idx, cell, loc = step
    #     # perform the kriging
    #     kv = k.krige( data, model, hs, bw, loc, 4 )
    #     # add the kriging estimate to the output
    #     M[cell[0],cell[1]] = kv
    #     # add the kriging estimate to a spatial location
    #     newdata = [ loc[0], loc[1], kv ]
    #     # add this new point to the data used for kriging
    #     data = np.vstack(( data, newdata ))
    # return M