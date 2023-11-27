import numpy as np
import random
from math import ceil, sqrt, isnan
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


def spiral_search(params: Params, debug=False):
    x = ceil(min(params.covar.range0[0] * params.neigh.wradius, params.nx))
    ss_X = np.linspace(-x, x, num=2 * x + 1)
    if params.ndim == 1:
        ss_dist = ss_X/params.covar.range0[0]
    if params.ndim >= 2:
        y = ceil(min(params.covar.range0[1] * params.neigh.wradius, params.ny))
        ss_Y = np.linspace(-y, y, num=2 * y + 1)
        ss_X, ss_Y = np.meshgrid(ss_X, ss_Y)
        ss_dist = np.sqrt((ss_X / params.covar.range0[0]) ** 2 + (ss_Y / params.covar.range0[1]) ** 2)
    if params.ndim >= 3:
        z = ceil(min(params.covar.range0[2] * params.neigh.wradius, params.nz))
        ss_Z = np.linspace(-z, z, num=2 * z + 1)
        ss_X, ss_Y, ss_Z = np.meshgrid(ss_X, ss_Y, ss_Z)
        ss_dist = np.sqrt((ss_X / params.covar.range0[0]) ** 2 + (ss_Y / params.covar.range0[1]) ** 2 + (ss_Z / params.covar.range0[2]) ** 2)

    ss_id_1 = ss_dist.flatten()
    ss_id_1 = np.argwhere(ss_id_1 <= params.neigh.wradius)
    ss_id_1 = ss_id_1.flatten()
    np.random.seed(params.seed_search)
    ss_id_1 = ss_id_1[np.random.permutation(len(ss_id_1))]

    if debug:
        ss_id_1_from_file = []
        with open('ss_id_1.txt') as f:
            for line in f:
                ss_id_1_from_file.append([int(x) for x in line.split()])
        ss_id_1 = np.array(ss_id_1_from_file).flatten()

    s = ss_dist.flatten()[ss_id_1]
    ss_id_2 = np.asarray(sorted(range(len(s)), key=lambda k: s[k]))
    ss_X_s = np.transpose(ss_X).flatten()[ss_id_1[ss_id_2]]
    ss_n = len(ss_X_s)
    ss_scale_s = np.ones(len(ss_id_2))
    if params.ndim == 2:
        ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
    if params.ndim == 3:
        ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
        ss_Z_s = np.transpose(ss_Z).flatten()[ss_id_1[ss_id_2]]

    ss_id = np.argwhere(ss_scale_s <= 1).T[0]
    if params.ndim == 1:
        ss_XY_s = ss_X_s[ss_id]
    elif params.ndim == 2:
        ss_XY_s = np.zeros((len(ss_Y_s), 2))
        for i in range(np.shape(ss_XY_s)[0]):
            ss_XY_s[i] = [ss_X_s[ss_id[i]], ss_Y_s[ss_id[i]]]
    elif params.ndim == 3:
        ss_XY_s = np.zeros((len(ss_Y_s), 3))
        for i in range(np.shape(ss_XY_s)[0]):
            ss_XY_s[i] = [ss_X_s[ss_id[i]], ss_Y_s[ss_id[i]], ss_Z_s[ss_id[i]]]
    else:
        raise ValueError('Only works in 1, 2 or 3D')

    return ss_XY_s


def map_2d(width, row, col):
    return width*row+col


def find_neighbors(params: Params, ss_XY_s, XY_i, Path, i_pt, nnc=False, debug=False):

    n = 0
    neigh_nn = np.zeros((params.neigh.nb, 1)) + np.nan
    NEIGH_1 = np.zeros((params.neigh.nb, 1)) + np.nan
    NEIGH_2 = np.zeros((params.neigh.nb, 1)) + np.nan
    NEIGH_3 = np.zeros((params.neigh.nb, 1)) + np.nan
    for nn in range(1, np.shape(ss_XY_s)[0]):
        ijt = XY_i[i_pt] + ss_XY_s[nn]
        ijt -= 1
        if nnc is True:

            if ijt[0] <= -1:
                ijt[0] += params.nx
            if ijt[1] <= -1:
                ijt[1] += params.ny
            if ijt[0] >= params.nx:
                ijt[0] -= params.nx
            if ijt[1] >= params.ny:
                ijt[1] -= params.ny

            if Path[int(ijt[0]), int(ijt[1])] < i_pt:
                neigh_nn[n] = nn
                NEIGH_1[n] = ijt[1] + 1
                NEIGH_2[n] = ijt[0] + 1
                n += 1
                if n >= params.neigh.nb:
                    break
        else:
            if params.ndim == 1:
                if ijt[0] > -1 and ijt[0] < params.nx:
                    if Path[int(ijt[0])] < i_pt:
                        neigh_nn[n] = nn
                        NEIGH_1[n] = ijt[0] + 1
                        n += 1
                        if n >= params.neigh.nb:
                            break
            if params.ndim == 2:
                if ijt[0] > -1 and ijt[0] < params.nx and ijt[1] > -1 and ijt[1] < params.ny:
                    if Path[int(ijt[0]), int(ijt[1])] < i_pt:
                        neigh_nn[n] = nn
                        NEIGH_1[n] = ijt[1] + 1
                        NEIGH_2[n] = ijt[0] + 1
                        n += 1
                        if n >= params.neigh.nb:
                            break
            if params.ndim == 3:
                if ijt[0] > -1 and ijt[0] < params.nx and ijt[1] > -1 and ijt[1] < params.ny and ijt[2] > -1 and ijt[
                    2] < params.nz:
                    if Path[int(ijt[0]), int(ijt[1]), int(ijt[2])] < i_pt:
                        neigh_nn[n] = nn
                        NEIGH_1[n] = ijt[2] + 1
                        NEIGH_2[n] = ijt[1] + 1
                        NEIGH_3[n] = ijt[0] + 1
                        n += 1
                        if n >= params.neigh.nb:
                            break

    if params.ndim == 1:
        return neigh_nn, n, NEIGH_1
    elif params.ndim == 2:
        return neigh_nn, n, NEIGH_1, NEIGH_2
    elif params.ndim == 3:
        return neigh_nn, n, NEIGH_1, NEIGH_2, NEIGH_3
    else:
        raise ValueError('Only works in 1, 2 or 3D')



def trad_sgs(params: Params, data):
    models_list = {
        'gaussian': model.gaussian,
        'exponential': model.exponential,
        'spherical': model.spherical
    }
    covfct = model.covariance(models_list.get(params.covar.model), (params.covar.range0[0], params.covar.c0))

    # creating grid
    if params.ndim == 1:
        X, grid = create_grid(params)
    elif params.ndim == 2:
        X, Y, grid = create_grid(params)
    elif params.ndim == 3:
        X, Y, Z, grid = create_grid(params)
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
        ss_XY_s = spiral_search(params=params)
        data_m = data
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
            if params.ndim == 1:
                neigh_nn, n, NEIGH_1 = find_neighbors(params, ss_XY_s, XY_i, Path, i_pt)
            elif params.ndim == 2:
                neigh_nn, n, NEIGH_1, NEIGH_2 = find_neighbors(params, ss_XY_s, XY_i, Path, i_pt)
            elif params.ndim == 3:
                neigh_nn, n, NEIGH_1, NEIGH_2, NEIGH_3 = find_neighbors(params, ss_XY_s, XY_i, Path, i_pt)

            statement = n == 0
            #statement = type(data_m) is list and not data_m
            if statement:
                Res = Res.flatten()
                Res[path[i_pt]] = U[i_pt] * np.sum([params.covar.c0])
                if params.ndim == 1:
                    data_m = [[int(XY_i[i_pt][0]), Res[path[i_pt]]]]
                elif params.ndim == 2:
                    data_m = [[int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), Res[path[i_pt]]]]
                elif params.ndim == 3:
                    data_m = [[int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), int(XY_i[i_pt][2]), Res[path[i_pt]]]]
                else:
                    raise ValueError('Only works in 1, 2 or 3D')
            else:
                # neigh_nn = neigh_nn[~np.isnan(neigh_nn)].astype(int)
                # n_datas = []
                # for i_nn in range(len(neigh_nn)):
                #     n_pt = XY_i[i_pt] + ss_XY_s[neigh_nn[i_nn]] - 1
                #     n_pt = n_pt.astype(int)
                #     if not len(n_datas):
                #         n_datas = [[n_pt[0], n_pt[1], Res[map_2d(params.nx, n_pt[0], n_pt[1])]]]
                #     else:
                #         n_data = [n_pt[0], n_pt[1], Res[map_2d(params.nx, n_pt[0], n_pt[1])]]
                #         n_datas = np.vstack((n_datas, n_data))

                est, kstd = k3d.simple(params, data_m, covfct, XY_i[i_pt].astype(int), nugget=params.covar.nugget,
                                       max_dist=params.neigh.wradius, N=params.neigh.nb)
                Res = Res.flatten()
                if est is None:
                    Res[path[i_pt]] = U[i_pt] * np.sum([params.covar.c0])
                else:
                    if isnan(kstd):
                        raise ValueError('kstd is NaN')
                    Res[path[i_pt]] = est + U[i_pt] * kstd

                # if i_pt > 2450:
                #     print(i_pt)
                #     print(est, kstd, Res[path[i_pt]])
                #     print(n)
                #     print(np.shape(n_datas))
                #     print(n_datas)
                #     print(ss_XY_s)

                if params.ndim == 1:
                    newdata = [int(XY_i[i_pt][0]), Res[path[i_pt]]]
                elif params.ndim == 2:
                    newdata = [int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), Res[path[i_pt]]]
                elif params.ndim == 3:
                    newdata = [int(XY_i[i_pt][0]), int(XY_i[i_pt][1]), int(XY_i[i_pt][2]), Res[path[i_pt]]]
                else:
                    raise ValueError('Only works in 1, 2 or 3D')
                data_m = np.vstack((data_m, newdata))

        if params.ndim == 1:
            Res = Res.reshape((params.nx))
        if params.ndim == 2:
            Res = Res.reshape((params.nx, params.ny))
        if params.ndim == 3:
            Res = Res.reshape((params.nx, params.ny, params.nz))

        if params.ndim == 1:
            Rest[:, i_real] = Res
        elif params.ndim == 2:
            Rest[:, :, i_real] = Res
        elif params.ndim == 3:
            Rest[:, :, :, i_real] = Res
        else:
            raise ValueError('Only works in 1, 2 or 3D')

    for m in range(np.shape(Rest)[-1]):
        if params.ndim == 1:
            Rest_means[m] = np.mean(Rest[:, m])
            Rest_std[m] = np.std(Rest[:, m])
        elif params.ndim == 2:
            Rest_means[m] = np.mean(Rest[:, :, m])
            Rest_std[m] = np.std(Rest[:, :, m])
        elif params.ndim == 3:
            Rest_means[m] = np.mean(Rest[:, :, :, m])
            Rest_std[m] = np.std(Rest[:, :, :, m])
        else:
            raise ValueError('Only works in 1, 2 or 3D')

    Rest = Rest + params.mean
    Rest_means = Rest_means + params.mean

    return Rest, Rest_means, Rest_std, grid, CY, U