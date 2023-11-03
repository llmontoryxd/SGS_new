import numpy as np
from math import ceil, sqrt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import kron, csr_matrix
from scipy.sparse.linalg import inv
from tqdm import tqdm
from models import Params


def sgs_trad(params: Params, debug=False, nnc=False, category=False):
    # creating grid
    X = np.linspace(1, params.nx, num=params.nx)
    Y = np.linspace(1, params.ny, num=params.ny)
    grid = np.zeros((len(X), len(Y), 2))
    for k in range(np.shape(grid)[2]):
        for j in range(np.shape(grid)[1]):
            for i in range(np.shape(grid)[0]):
                grid[i, j, :] = [X[i], Y[j]]
    grid = grid.reshape(np.shape(grid)[0]*np.shape(grid)[1], 2)
    X, Y = np.meshgrid(X, Y)
    print(params.nnc)



    # spiral search of neighbours
    x = ceil(min(params.covar.range[1]*params.neigh.wradius, params.nx))
    y = ceil(min(params.covar.range[0]*params.neigh.wradius, params.ny))

    ss_Y = np.linspace(-y, y, num=2*y+1)
    ss_X = np.linspace(-x, x, num=2*x+1)
    ss_X, ss_Y = np.meshgrid(ss_X, ss_Y)

    ss_dist = np.sqrt((ss_X/params.covar.range[0])**2 + (ss_Y/params.covar.range[1])**2)

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
    ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
    ss_n = len(ss_X_s)
    ss_scale_s = np.ones(len(ss_id_2))


    # lookup table
    if params.neigh.lookup:
        a0_h = np.zeros((len(ss_X_s), 2))
        ss_vec = np.zeros((len(ss_X_s), 2))
        for i in range(np.shape(a0_h)[0]):
            ss_vec[i] = [ss_X_s[i], ss_Y_s[i]]
        a0_h = np.sqrt(np.sum(np.matmul(ss_vec, params.covar.cx)**2, axis=1))
        ab_h = squareform(pdist(np.matmul(ss_vec, params.covar.cx)))

        ss_a0_C = np.kron(params.covar.g(a0_h), params.covar.c0)
        ss_ab_C = kron(params.covar.g(ab_h), params.covar.c0).toarray()


    # realization loop
    Rest = np.zeros((params.nx, params.ny, params.m)) + np.nan
    Rest_means = np.zeros(params.m)
    Rest_std = np.zeros(params.m)
    LambdaM = np.zeros((params.nx*params.ny, params.nx*params.ny, params.m))
    CY = np.zeros((params.nx*params.ny, params.nx*params.ny, params.m))
    for i_real in tqdm(range(np.shape(Rest)[2])):
        Res = np.zeros((params.nx, params.ny)) + np.nan

        # generation of the path
        Path = np.zeros((params.nx, params.ny)) + np.nan
        Path = Path.flatten()
        path = np.zeros((params.nx*params.ny, 1)) + np.nan
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
        Path = Path.reshape((params.nx, params.ny))
        ds = 1
        nb = [len(id)]
        start = [0, nb]
        sn = 1
        Yf = np.transpose(Y).flatten()
        Xf = np.transpose(X).flatten()
        XY_i = np.zeros((len(Xf), 2))
        for i in range(np.shape(XY_i)[0]):
            XY_i[i] = [Xf[path[i]], Yf[path[i]]]

        np.random.seed(params.seed_U)
        U = np.random.randn(params.nx, params.ny).flatten()

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
            ss_XY_s = np.zeros((len(ss_Y_s), 2))
            for i in range(np.shape(ss_XY_s)[0]):
                ss_XY_s[i] = [ss_X_s[ss_id[i]], ss_Y_s[ss_id[i]]]

            if params.neigh.lookup:
                ss_a0_C_s = ss_a0_C[ss_id]
                ss_ab_C_s = ss_ab_C[ss_id]

            for i_pt in range(start[i_scale], nb[i_scale]+start[i_scale]):
                n = 0
                neigh_nn = np.zeros((params.neigh.nb, 1)) + np.nan
                neigh_nn_for_id = np.zeros((params.neigh.nb, 3)) + np.nan
                NEIGH_1 = np.zeros((params.neigh.nb, 1)) + np.nan
                NEIGH_2 = np.zeros((params.neigh.nb, 1)) + np.nan
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
                        if ijt[0] > -1 and ijt[1] > -1 and ijt[0] < params.nx and ijt[1] < params.ny:
                            if Path[int(ijt[0]), int(ijt[1])] < i_pt:
                                neigh_nn[n] = nn
                                NEIGH_1[n] = ijt[1]+1
                                NEIGH_2[n] = ijt[0]+1
                                neigh_nn_for_id[n, :] = [nn, ijt[1] + 1, ijt[0] + 1]
                                n += 1
                                if n >= params.neigh.nb:
                                    break
                if n == 0:
                    Res = Res.flatten()
                    Res[path[i_pt]] = U[i_pt]*np.sum([params.covar.c0])
                    NEIGH = []
                else:
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
                        left = np.array([0, 0])
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

                    NEIGH = NEIGH_1 + (NEIGH_2-1)*params.ny
                    NEIGH = NEIGH[:n].flatten()
                    right = []
                    Res = Res.flatten()
                    for el in NEIGH:
                        right.append(Res[int(el)-1])

                    if params.calc_frob:
                        LambdaM[path[i_pt], neigh_id, i_real] = LAMBDA / sqrt(S)
                        LambdaM[path[i_pt], path[i_pt], i_real] = -1 / sqrt(S)
                    Res[path[i_pt]] = np.matmul(np.transpose(LAMBDA), right) + U[i_pt]*sqrt(S)

        Res = Res.reshape((params.nx, params.ny))

        if params.calc_frob:
            CY[:, :, i_real] = np.linalg.lstsq(csr_matrix(LambdaM[:, :, i_real]).toarray(),
                                     np.transpose(np.linalg.pinv(csr_matrix(LambdaM[:, :, i_real]).toarray())),
                         rcond=None)[0]
        Rest[:, :, i_real] = Res

    if category:
        for k in range(np.shape(Rest)[2]):
            for j in range(np.shape(Rest)[1]):
                for i in range(np.shape(Rest)[0]):
                    el = Rest[i, j, k]
                    if el > params.cat_threshold:
                        Rest[i, j, k] = 1
                    else:
                        Rest[i, j, k] = 0
    Rest = np.transpose(Rest, (1, 0, 2))

    for m in range(np.shape(Rest)[2]):
        Rest_means[m] = np.mean(Rest[:, :, m])
        Rest_std[m] = np.std(Rest[:, :, m])
    if not category:
        Rest = Rest + params.mean
        Rest_means = Rest_means + params.mean

    return Rest, Rest_means, Rest_std, grid, CY, U







