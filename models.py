from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from math import cos, sin, pi
from typing import Callable


class Params:
    nx: int
    ny: int
    m: int
    mean: float
    covar: Covar
    neigh: Neigh
    seed_search = None
    seed_path = None
    seed_U = None

    def __init__(self, nx, ny, m, mean, covar, neigh):
        self.nx = nx
        self.ny = ny
        self.m = m
        self.mean = mean
        self.covar = covar
        self.neigh = neigh


class Covar:
    model: str
    range0: list[float]
    range: list
    azimuth: list
    c0: float
    cx: ArrayLike
    alpha: float
    intvario: list
    g: float
    models_list = ['gaussian', 'exponential', 'spherical', 'hyperbolic']

    def __init__(self, model, range0, azimuth, c0, alpha):
        self.model = model
        self.range0 = range0
        self.azimuth = azimuth
        self.c0 = c0
        self.alpha = alpha
        assert self.model in self.models_list
        self.__call__()

    def __call__(self):
        match self.model:
            case 'gaussian':
                self.intvario = [.58]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: np.exp(-h**2/self.alpha**2)
            case 'exponential':
                self.intvario = [.41]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: np.exp(-h/self.alpha)
            case 'spherical':
                self.intvario = [1.3]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: 1-3/2*np.minimum(h/self.alpha, 1)+1/2*np.minimum(h/self.alpha, 1)**3
            case 'hyperbolic':
                self.intvario = [.2, .05]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: 1/(1+h)
            case _:
                raise ValueError('Your model is not supported yet')

        self.range = np.asarray(self.range0) * self.intvario[-1]

        if len(self.range) == 1 or len(self.azimuth) == 0:
            self.cx = 1/np.diag(self.range[0])
        elif len(self.range) == 2:
            ang = self.azimuth[0]
            cang = cos(ang/180*pi)
            sang = sin(ang/180*pi)
            rot = [[cang, -sang],
                   [sang, cang]]
            self.cx = rot/np.diag(self.range)

        self.cx = np.nan_to_num(self.cx)


class Neigh:
    wradius: int
    lookup: bool
    nb: int

    def __init__(self, wradius, lookup, nb):
        self.wradius = wradius
        self.lookup = lookup
        self.nb = nb


