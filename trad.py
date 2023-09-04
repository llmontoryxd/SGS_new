import random

import numpy as np
from typing import Type
from math import ceil, sqrt
from scipy.spatial.distance import pdist
from models import Params


def sgs_trad(params: Params, debug=False):
    # creating grid
    X = np.linspace(1, params.nx, num=params.nx)
    Y = np.linspace(1, params.ny, num=params.ny)
    X, Y = np.meshgrid(X, Y)



    # spiral search of neighbours
    x = ceil(min(params.covar.range[1]*params.neigh.wradius, params.nx))
    y = ceil(min(params.covar.range[0]*params.neigh.wradius, params.ny))
    #print(x, y)

    ss_Y = np.linspace(-y, y, num=2*y+1)
    ss_X = np.linspace(-x, x, num=2*x+1)
    ss_X, ss_Y = np.meshgrid(ss_X, ss_Y)
    #print(ss_X, ss_Y)

    ss_dist = np.sqrt((ss_X/params.covar.range[0])**2 + (ss_Y/params.covar.range[1])**2)
    #print(ss_dist)

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
    #print(np.shape(ss_id_1), np.shape(ss_id_2))
    #print(ss_id_1)
    #print(ss_id_2)
    #print(ss_X.flatten()[1])
    #print(ss_X.flatten())
    #print(ss_id_1[ss_id_2])
    #print(ss_X, ss_Y)
    ss_X_s = np.transpose(ss_X).flatten()[ss_id_1[ss_id_2]]
    ss_Y_s = np.transpose(ss_Y).flatten()[ss_id_1[ss_id_2]]
    #print(ss_X_s)
    #print(ss_Y_s)
    ss_n = len(ss_X_s)
    print(ss_n)
    ss_scale_s = np.ones(len(ss_id_2))


    # covariance lookup table
    ...

    # realization loop
    Rest = np.zeros((params.nx, params.ny, params.m)) + np.nan
    for i_real in range(np.shape(Rest)[2]):
        Res = np.zeros((params.nx, params.ny)) + np.nan

        # generation of the path
        Path = np.zeros((params.nx, params.ny)) + np.nan
        Path = Path.flatten()
        path = np.zeros((params.nx*params.ny, 1)) + np.nan
        np.random.seed(params.seed_path)
        id = np.argwhere(np.isnan(Path.flatten())).T[0]
        #print(id)
        path = id[np.random.permutation(len(id))]

        if debug:
            path_from_file = []
            with open('path.txt') as f:
                for line in f:
                    path_from_file.append([int(x) for x in line.split()])
            path = np.array(path_from_file).flatten()

        #print(path)
        Path[path] = range(len(id))
        Path = Path.reshape((params.nx, params.ny))
        #Path = np.transpose(Path)
        #print(Path)
        ds = 1
        nb = [len(id)]
        start = [0, nb]
        sn = 1
        Yf = np.transpose(Y).flatten()
        Xf = np.transpose(X).flatten()
        XY_i = np.zeros((len(Xf), 2))
        for i in range(np.shape(XY_i)[0]):
            XY_i[i] = [Xf[path[i]], Yf[path[i]]]
        #XY_i = [Y.flatten()[path], X.flatten()[path]]
        #print(XY_i)

        np.random.seed(params.seed_U)
        U = np.random.randn(params.nx, params.ny).flatten()

        if debug:
            U_from_file = []
            with open('U.txt') as f:
                for line in f:
                    U_from_file.append([float(x) for x in line.split()])
            U = np.array(U_from_file).flatten()
            U = U.reshape((params.nx, params.ny))
            print(U)
            U = U.flatten()

        #print(U)
        #print(Path[25, 49])
        #print(path)
        for i_scale in range(sn):
            ss_id = np.argwhere(ss_scale_s <= i_scale+1).T[0]
            ss_XY_s = np.zeros((len(ss_Y_s), 2))
            for i in range(np.shape(ss_XY_s)[0]):
                ss_XY_s[i] = [ss_X_s[ss_id[i]], ss_Y_s[ss_id[i]]]
            #print(ss_XY_s)

            for i_pt in range(start[i_scale], nb[i_scale]+start[i_scale]):
                n = 0
                neigh_nn = np.zeros((params.neigh.nb, 1)) + np.nan
                NEIGH_1 = np.zeros((params.neigh.nb, 1)) + np.nan
                NEIGH_2 = np.zeros((params.neigh.nb, 1)) + np.nan
                for nn in range(1, np.shape(ss_XY_s)[0]):
                    ijt = XY_i[i_pt] + ss_XY_s[nn]
                    #print(ijt)
                    if ijt[0] > 0 and ijt[1] > 0 and ijt[0] <= params.nx and ijt[1] <= params.ny:
                        #print(Path[int(ijt[1])-1, int(ijt[0])-1])
                        #print(i_pt)
                        if Path[int(ijt[0])-1, int(ijt[1])-1] < i_pt:
                            #print(ijt)
                            #print(Path[int(ijt[1])-1, int(ijt[0])-1])
                            if i_pt == 13:
                                ...
                                #print(ijt)
                                #print(Path[int(ijt[1])-1, int(ijt[0])-1])
                            #print('-------------------')
                            neigh_nn[n] = nn
                            NEIGH_1[n] = ijt[1]
                            NEIGH_2[n] = ijt[0]
                            n += 1
                            if n >= params.neigh.nb:
                                break
                #print(n)
                if n == 0:
                    ...
                    #print(path[i_pt])
                    Res = Res.flatten()
                    #print(np.sum([params.covar.c0]))
                    Res[path[i_pt]] = U[i_pt]*np.sum([params.covar.c0])
                    NEIGH = []
                else:
                    left = np.array([0, 0])
                    neigh_nn = neigh_nn[~np.isnan(neigh_nn)]
                    for idn in neigh_nn:
                        left = np.vstack((left, ss_XY_s[int(idn)]))

                    D = pdist(np.matmul(left, params.covar.cx))
                    C = params.covar.g(D)
                    #C = np.round(C, 4)

                    if n == 1:
                        a0_C = C
                        ab_C = [[1]]
                    else:
                        a0_C = np.transpose(C[:n])
                        #if i_pt < 10:
                        #    print(n, C, a0_C)
                        ab_C = np.diag(np.ones(n))*0.5
                        low_t = np.tril(np.ones((n, n)), -1)
                        n_next = n
                        for j in range(np.shape(low_t)[0]):
                            for i in range(np.shape(low_t)[1]):
                                if low_t[i, j] == 1:
                                    low_t[i, j] = C[n_next]
                                    n_next += 1
                        #print(low_t)
                        ab_C += low_t
                        ab_C += np.transpose(ab_C)
                        #ab_C = np.round(ab_C, 4)


                    LAMBDA = np.linalg.lstsq(ab_C, a0_C, rcond=None)[0]
                    S = np.sum([params.covar.c0]) - np.matmul(np.transpose(LAMBDA), a0_C)
                    if i_pt < 10:
                        ...
                        #print(LAMBDA)
                        #print(ab_C, a0_C)
                        #print(sqrt(S))
                    NEIGH = NEIGH_1 + (NEIGH_2-1)*params.ny
                    NEIGH = NEIGH[:n].flatten()
                    right = []
                    Res = Res.flatten()
                    for el in NEIGH:
                        right.append(Res[int(el)-1])
                    #print(S)
                    #print(n)
                    #print(C)
                    #print(ab_C)
                    #print('---------------------')
                    Res[path[i_pt]] = np.matmul(np.transpose(LAMBDA), right) + U[i_pt]*sqrt(S)
                    #print(Res)
                    #print(f'{n}, {i_pt}\n')
                    #print(path[i_pt])
                    #print(right)
                    if i_pt < 10:
                        ...
                        #print(LAMBDA)
                        #print(C)
                        #print(ab_C, a0_C)
                        #print(NEIGH)
                        #print(right)
                        #print(Res[int(NEIGH[0])])
                        #print(Res[path[i_pt]])
                        #print(Res)
                        #print(S)
                        #print('---------------------')
        #print(i_real)
        Res = Res.reshape((params.nx, params.ny))
        #print(Res)
        #print(np.shape(Res))
        Rest[:, :, i_real] = Res
        #print(np.shape(Rest))

    Rest = np.transpose(Rest, (1, 0, 2))
    return Rest+params.mean







