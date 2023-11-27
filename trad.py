import random

import numpy as np
from math import ceil, sqrt, isnan
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import inv
from tqdm import tqdm
from models import Params

import k3d
import model


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
    x = ceil(min(params.covar.range[0] * params.neigh.wradius, params.nx))
    print(x)
    ss_X = np.linspace(-x, x, num=2 * x + 1)
    if params.ndim == 1:
        ss_dist = ss_X/params.covar.range[0]
    if params.ndim >= 2:
        y = ceil(min(params.covar.range[1] * params.neigh.wradius, params.ny))
        ss_Y = np.linspace(-y, y, num=2 * y + 1)
        ss_X, ss_Y = np.meshgrid(ss_X, ss_Y)
        ss_dist = np.sqrt((ss_X / params.covar.range[0]) ** 2 + (ss_Y / params.covar.range[1]) ** 2)
    if params.ndim >= 3:
        z = ceil(min(params.covar.range[2] * params.neigh.wradius, params.nz))
        ss_Z = np.linspace(-z, z, num=2 * z + 1)
        ss_X, ss_Y, ss_Z = np.meshgrid(ss_X, ss_Y, ss_Z)
        ss_dist = np.sqrt((ss_X / params.covar.range[0]) ** 2 + (ss_Y / params.covar.range[1]) ** 2 + (ss_Z / params.covar.range[2]) ** 2)

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
    if params.ndim == 1:
        return ss_X_s, ss_n, ss_scale_s
    if params.ndim == 2:
        ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
        return ss_X_s, ss_Y_s, ss_n, ss_scale_s
    if params.ndim == 3:
        ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
        ss_Z_s = np.transpose(ss_Z).flatten()[ss_id_1[ss_id_2]]
        return ss_X_s, ss_Y_s, ss_Z_s, ss_n, ss_scale_s


def get_lookup_table(params: Params, x_s, y_s, z_s):
    if params.ndim == 1:
        ss_vec = x_s
    if params.ndim == 2:
        ss_vec = np.zeros((len(x_s), 2))
        for i in range(np.shape(ss_vec)[0]):
            ss_vec[i] = [x_s[i], y_s[i]]
    if params.ndim == 3:
        ss_vec = np.zeros((len(x_s), 3))
        for i in range(np.shape(ss_vec)[0]):
            ss_vec[i] = [x_s[i], y_s[i], z_s[i]]

    print(np.shape(ss_vec))
    a0_h = np.sqrt(np.sum(np.matmul(ss_vec, params.covar.cx) ** 2, axis=1))
    ab_h = squareform(pdist(np.matmul(ss_vec, params.covar.cx)))
    #ab_h = cdist(np.matmul(ss_vec, params.covar.cx), np.matmul(ss_vec, params.covar.cx))
    print(np.shape(ab_h))

    ss_a0_C = np.kron(params.covar.g(a0_h), params.covar.c0)
    ss_ab_C = kron(params.covar.g(ab_h), params.covar.c0).toarray()
    print(np.shape(ss_ab_C))

    return ss_a0_C, ss_ab_C

def map_2d(width, row, col):
    return width*row+col

def cont_to_indicator(threshold: float):
    value = np.random.uniform(low=0.0, high=1.0, size=1)
    if value <= threshold:
        return 1
    else:
        return 0

def cont_to_indicator_arr(threshold: float, arr):
    for i in range(len(arr)):
        if arr[i] <= threshold:
            arr[i] = 1
        else:
            arr[i] = 0

    return arr


def sgs_trad(params: Params, debug=False, nnc=False, category=False):
    models_list = {
        'gaussian': model.gaussian,
        'exponential': model.exponential,
        'spherical': model.spherical
    }

    # creating grid
    if params.ndim == 1:
        X, grid = create_grid(params)
    elif params.ndim == 2:
        X, Y, grid = create_grid(params)
    elif params.ndim == 3:
        X, Y, Z, grid = create_grid(params)
        #print(X, Y, Z)
    else:
        raise ValueError('Only works in 1, 2 or 3D')



    # spiral search of neighbours
    if params.ndim == 1:
        ss_X_s, ss_n, ss_scale_s = spiral_search(params, debug)
    elif params.ndim == 2:
        ss_X_s, ss_Y_s, ss_n, ss_scale_s = spiral_search(params, debug)
    elif params.ndim == 3:
        ss_X_s, ss_Y_s, ss_Z_s, ss_n, ss_scale_s = spiral_search(params, debug)
        #print(ss_X_s, ss_Y_s, ss_Z_s)
    else:
        raise ValueError('Only works in 1, 2 or 3D')


    # lookup table
    if params.neigh.lookup:
        if params.ndim == 1:
            ss_a0_C, ss_ab_C = get_lookup_table(params, ss_X_s, [], [])
        elif params.ndim == 2:
            ss_a0_C, ss_ab_C = get_lookup_table(params, ss_X_s, ss_Y_s, [])
        elif params.ndim == 3:
            ss_a0_C, ss_ab_C = get_lookup_table(params, ss_X_s, ss_Y_s, ss_Z_s)
            # print(ss_X_s, ss_Y_s, ss_Z_s)
        else:
            raise ValueError('Only works in 1, 2 or 3D')


    # Output init
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
    if params.calc_frob:
        LambdaM = np.zeros((params.nx * params.ny, params.nx * params.ny, params.m))
        CY = np.zeros((params.nx * params.ny, params.nx * params.ny, params.m))
    else:
        CY = None



    # Main loop
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

        if debug:
            path_from_file = []
            with open('path.txt') as f:
                for line in f:
                    path_from_file.append([int(x) for x in line.split()])
            path = np.array(path_from_file).flatten()

        Path[path] = range(len(id))
        print(Path[path])
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
        ds = 1
        nb = [len(id)]
        start = [0, nb]
        sn = 1
        print(Path)
        print(path)


        # Get U
        np.random.seed(params.seed_U)
        if params.ndim == 1:
            if category:
                U = np.random.uniform(low=0.0, high=1.0, size=params.nx).flatten()
                U = cont_to_indicator_arr(params.cat_threshold, U)
            else:
                U = np.random.randn(params.nx).flatten()
        elif params.ndim == 2:
            if category:
                U = np.random.uniform(low=0.0, high=1.0, size=(params.nx, params.ny)).flatten()
                U = cont_to_indicator_arr(params.cat_threshold, U)
            else:
                U = np.random.randn(params.nx, params.ny).flatten()
        elif params.ndim == 3:
            if category:
                U = np.random.uniform(low=0.0, high=1.0, size=(params.nx, params.ny, params.nz)).flatten()
                U = cont_to_indicator_arr(params.cat_threshold, U)
            else:
                U = np.random.randn(params.nx, params.ny, params.nz).flatten()
        else:
            raise ValueError('Only works in 1, 2 or 3D')

        if debug:
            U_from_file = []
            with open('U.txt') as f:
                for line in f:
                    U_from_file.append([float(x) for x in line.split()])
            U = np.array(U_from_file).flatten()
            U = U.reshape((params.nx, params.ny))
            U = U.flatten()

        for i_scale in range(sn):
            ss_id = np.argwhere(ss_scale_s <= i_scale+1).T[0]
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

            if params.neigh.lookup:
                ss_a0_C_s = ss_a0_C[ss_id]
                ss_ab_C_s = ss_ab_C[ss_id]

            for i_pt in tqdm(range(start[i_scale], nb[i_scale]+start[i_scale])):
                n = 0
                neigh_nn = np.zeros((params.neigh.nb, 1)) + np.nan
                if params.ndim == 1:
                    neigh_nn_for_id = np.zeros((params.neigh.nb, 2)) + np.nan
                elif params.ndim == 2:
                    neigh_nn_for_id = np.zeros((params.neigh.nb, 3)) + np.nan
                elif params.ndim == 3:
                    neigh_nn_for_id = np.zeros((params.neigh.nb, 4)) + np.nan
                else:
                    raise ValueError('Only works in 1, 2 or 3D')
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
                            neigh_nn_for_id[n, :] = [nn, ijt[1] + 1, ijt[0] + 1]
                            n += 1
                            if n >= params.neigh.nb:
                                break
                    else:
                        if params.ndim == 1:
                            if ijt[0] > -1 and ijt[0] < params.nx:
                                if Path[int(ijt[0])] < i_pt:
                                    neigh_nn[n] = nn
                                    NEIGH_1[n] = ijt[0] + 1
                                    neigh_nn_for_id[n, :] = [nn, ijt[0] + 1]
                                    n += 1
                                    if n >= params.neigh.nb:
                                        break
                        if params.ndim == 2:
                            if ijt[0] > -1 and ijt[0] < params.nx and ijt[1] > -1 and ijt[1] < params.ny:
                                    if Path[int(ijt[0]), int(ijt[1])] < i_pt:
                                        neigh_nn[n] = nn
                                        NEIGH_1[n] = ijt[1]+1
                                        NEIGH_2[n] = ijt[0]+1
                                        neigh_nn_for_id[n, :] = [nn, ijt[1] + 1, ijt[0] + 1]
                                        n += 1
                                        if n >= params.neigh.nb:
                                            break
                        if params.ndim == 3:
                            if ijt[0] > -1 and ijt[0] < params.nx and ijt[1] > -1 and ijt[1] < params.ny and ijt[2] > -1 and ijt[2] < params.nz:
                                    if Path[int(ijt[0]), int(ijt[1]), int(ijt[2])] < i_pt:
                                        neigh_nn[n] = nn
                                        NEIGH_1[n] = ijt[2]+1
                                        NEIGH_2[n] = ijt[1]+1
                                        NEIGH_3[n] = ijt[0] + 1
                                        neigh_nn_for_id[n, :] = [nn, ijt[2] + 1, ijt[1] + 1, ijt[0] + 1]
                                        n += 1
                                        if n >= params.neigh.nb:
                                            break
                if n == 0:
                    Res = Res.flatten()
                    Res[path[i_pt]] = U[i_pt]*np.sum([params.covar.c0])
                    if category:
                        Res[path[i_pt]] = cont_to_indicator(Res[path[i_pt]])
                    NEIGH = []
                else:
                    if params.ndim == 2:
                        neigh_id = np.matmul(neigh_nn_for_id[:n, 1:3] - 1, np.transpose([1, params.ny]))
                        neigh_id = neigh_id.astype(int)
                    if params.neigh.lookup:
                        neigh_nn = neigh_nn[~np.isnan(neigh_nn)].astype(int)
                        a0_C = ss_a0_C_s[neigh_nn[:n]]
                        neigh_nn_len = len(neigh_nn[:n])
                        ab_C = np.zeros((neigh_nn_len, neigh_nn_len))
                        for i in range(neigh_nn_len):
                            ab_C[i] = ss_ab_C_s[neigh_nn[i], neigh_nn[:n]]

                        if ab_C.ndim == 1:
                            ab_C = [ab_C]
                    else:
                        left = np.zeros(params.ndim)
                        neigh_nn = neigh_nn[~np.isnan(neigh_nn)]
                        for idn in neigh_nn:
                            left = np.vstack((left, ss_XY_s[int(idn)]))

                        D = pdist(np.matmul(left, params.covar.cx))
                        C = params.covar.g(D)

                        if n == 1:
                            a0_C = C
                            ab_C = [[1]]
                        else:
                            a0_C = np.transpose(C[:n])
                            ab_C = np.diag(np.ones(n))*0.5
                            low_t = np.tril(np.ones((n, n)), -1)
                            n_next = n
                            for j in range(np.shape(low_t)[0]):
                                for i in range(np.shape(low_t)[1]):
                                    if low_t[i, j] == 1:
                                        low_t[i, j] = C[n_next]
                                        n_next += 1
                            ab_C += low_t
                            ab_C += np.transpose(ab_C)

                    LAMBDA = np.linalg.lstsq(ab_C, a0_C, rcond=None)[0]
                    S = np.sum([params.covar.c0]) - np.matmul(np.transpose(LAMBDA), a0_C)
                    #S = max(S, 0.0)
                    # print(S)

                    if params.ndim == 1:
                        NEIGH = NEIGH_1
                    if params.ndim == 2:
                        NEIGH = NEIGH_1 + (NEIGH_2 - 1) * params.ny
                    if params.ndim == 3:
                        NEIGH = NEIGH_1 + ((NEIGH_2 - 1) * params.ny + (NEIGH_3 - 1) - 1) * params.nz
                    NEIGH = NEIGH[:n].flatten()
                    right = []
                    Res = Res.flatten()
                    for el in NEIGH:
                        right.append(Res[int(el) - 1])

                    if params.calc_frob:
                        LambdaM[path[i_pt], neigh_id, i_real] = LAMBDA / sqrt(S)
                        LambdaM[path[i_pt], path[i_pt], i_real] = -1 / sqrt(S)
                    Res[path[i_pt]] = np.matmul(np.transpose(LAMBDA), right) + U[i_pt] * sqrt(S)
                    if category:
                        Res[path[i_pt]] = cont_to_indicator(Res[path[i_pt]])

        if params.ndim == 1:
            Res = Res.reshape((params.nx))
        if params.ndim == 2:
            Res = Res.reshape((params.nx, params.ny))
        if params.ndim == 3:
            Res = Res.reshape((params.nx, params.ny, params.nz))

        if params.calc_frob:
            CY[:, :, i_real] = np.linalg.lstsq(csr_matrix(LambdaM[:, :, i_real]).toarray(),
                                     np.transpose(np.linalg.pinv(csr_matrix(LambdaM[:, :, i_real]).toarray())),
                         rcond=None)[0]



        if params.ndim == 1:
            Rest[:, i_real] = Res
        elif params.ndim == 2:
            Rest[:, :, i_real] = Res
        elif params.ndim == 3:
            Rest[:, :, :, i_real] = Res
        else:
            raise ValueError('Only works in 1, 2 or 3D')

    Rest = np.transpose(Rest, (1, 0, 2))
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

    if not category:
        Rest = Rest + params.mean
        Rest_means = Rest_means + params.mean

    return Rest, Rest_means, Rest_std, grid, CY, U







