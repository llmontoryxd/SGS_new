from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from math import cos, sin, pi
from typing import Callable


class Config:
    model: str
    ndim: int
    range0: list[float]
    azimuth: list
    c0: float
    alpha: float
    wradius: int
    lookup: bool
    nb: int
    nx: int
    ny: int
    nz: int
    m: int
    mean: float
    nnc: bool
    category: bool
    cat_threshold: float
    debug: bool
    calc_frob: bool
    seed_search: str | None
    seed_path: str | None
    seed_U: str | None
    cutoff: float
    bins: int
    show: bool
    save: bool
    show_NNC: bool
    mode: str | None
    savefilename: str

    def __init__(self, filename):
        self.range0 = []
        self.azimuth = []
        self._read(filename)

    def _read(self, filename):
        with open(filename, 'r') as f:
            contents = f.readlines()
            for content in contents:
                content = content.split()
                if len(content) == 0:
                    continue
                match content[0]:
                    case 'model':
                        self.model = content[1]
                    case 'ndim':
                        self.ndim = int(content[1])
                    case 'range':
                        arg_nums = len(content) - 1
                        for arg_num in range(arg_nums):
                            self.range0.append(float(content[arg_num+1]))
                    case 'azimuth':
                        arg_nums = len(content) - 1
                        for arg_num in range(arg_nums):
                            self.azimuth.append(float(content[arg_num + 1]))
                    case 'c0':
                        self.c0 = float(content[1])
                    case 'alpha':
                        self.alpha = float(content[1])
                    case 'wradius':
                        self.wradius = int(content[1])
                    case 'lookup':
                        self.lookup = bool(int(content[1]))
                    case 'nb':
                        self.nb = int(content[1])
                    case 'nx':
                        self.nx = int(content[1])
                    case 'ny':
                        self.ny = int(content[1])
                    case 'nz':
                        self.nz = int(content[1])
                    case 'm':
                        self.m = int(content[1])
                    case 'mean':
                        self.mean = float(content[1])
                    case 'nnc':
                        self.nnc = bool(int(content[1]))
                    case 'category':
                        self.category = bool(int(content[1]))
                    case 'cat_threshold':
                        if len(content) == 1:
                            self.cat_threshold = 0
                        else:
                            self.cat_threshold = float(content[1])
                    case 'debug':
                        self.debug = bool(int(content[1]))
                    case 'calc_frob':
                        self.calc_frob = bool(int(content[1]))
                    case 'seed_search':
                        if len(content) == 1:
                            self.seed_search = None
                        else:
                            self.seed_search = content[1]
                    case 'seed_path':
                        if len(content) == 1:
                            self.seed_path = None
                        else:
                            self.seed_path = content[1]
                    case 'seed_U':
                        if len(content) == 1:
                            self.seed_U = None
                        else:
                            self.seed_U = content[1]
                    case 'cutoff':
                        self.cutoff = float(content[1])
                    case 'bins':
                        self.bins = int(content[1])
                    case 'show':
                        self.show = bool(int(content[1]))
                    case 'save':
                        self.save = bool(int(content[1]))
                    case 'savefilename':
                        self.savefilename = content[1]
                    case 'show_NNC':
                        self.show_NNC = bool(int(content[1]))
                    case 'mode':
                        if len(content) == 1:
                            self.mode = None
                        else:
                            self.mode = content[1]



class Params:
    ndim: int
    nx: int
    ny: int
    nz: int
    m: int
    mean: float
    covar: Covar
    neigh: Neigh
    nnc: bool
    category: bool
    cat_threshold: float
    debug: bool
    calc_frob: bool
    seed_search: str | None
    seed_path: str | None
    seed_U: str | None

    def __init__(self, ndim, nx, ny, nz, m, mean, covar, neigh, nnc=False,
                 category=False, cat_threshold=0.225,
                 debug=False, calc_frob=False, seed_search=None, seed_path=None, seed_U=None):
        self.ndim = ndim
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.m = m
        self.mean = mean
        self.covar = covar
        self.neigh = neigh
        self.nnc = nnc
        self.category = category
        self.debug = debug
        self.calc_frob = calc_frob
        self.cat_threshold = cat_threshold
        self.seed_search = seed_search
        self.seed_path = seed_path
        self.seed_U = seed_U


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
        match self.model:
            case 'gaussian':
                self.intvario = [.58]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: np.exp(-h**2)
            case 'exponential':
                self.intvario = [.41]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: np.exp(-h)
            case 'spherical':
                self.intvario = [1.3]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: 1-3/2*np.minimum(h, 1)+1/2*np.minimum(h, 1)**3
            case 'hyperbolic':
                self.intvario = [.2, .05]
                self.g: Callable[[ArrayLike], ArrayLike] = lambda h: 1/(1+h)
            case _:
                raise ValueError('Your model is not supported yet')

        self.range = np.asarray(self.range0) * self.intvario[-1]

        if len(self.range) == 1 or len(self.azimuth) == 0:
            self.cx = 1/np.diag([self.range[0]])
        elif len(self.range) == 2:
            ang = self.azimuth[0]
            cang = cos(ang/180*pi)
            sang = sin(ang/180*pi)
            rot = [[cang, -sang],
                   [sang, cang]]
            self.cx = rot/np.diag(self.range)
        elif len(self.range) == 3:
            theta = self.azimuth[0]
            phi = self.azimuth[1]
            ctheta = cos(theta/180*pi)
            stheta = sin(theta/180*pi)
            cphi = cos(phi/180*pi)
            sphi = sin(phi/180*pi)

            rot = [[ctheta*cphi, -stheta, -ctheta*sphi],
                   [stheta*cphi, ctheta, -stheta*sphi],
                   [sphi, 0, cphi]]

            self.cx = rot/np.diag(self.range)

        self.cx = np.nan_to_num(self.cx)
        print(self.cx)


class Neigh:
    wradius: int
    lookup: bool
    nb: int

    def __init__(self, wradius, lookup, nb):
        self.wradius = wradius
        self.lookup = lookup
        self.nb = nb


class Plot:
    cutoff: float
    bins: int
    show: bool
    save: bool
    show_NNC: bool
    mode: str | None
    savefilename: str

    def __init__(self, cutoff, bins, show, save, show_NNC, mode, savefilename):
        self.cutoff = cutoff
        self.bins = bins
        self.show = show
        self.save = save
        self.show_NNC = show_NNC
        self.mode = mode
        self.savefilename = savefilename


