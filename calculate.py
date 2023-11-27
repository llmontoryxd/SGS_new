import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

from trad import sgs_trad
from sgs import trad_sgs
from models import Params, Covar, Neigh, Config, Plot
from plot import plot
from frob import frob


def calculate(configfilename):
    config = Config(configfilename)

    covar = Covar(model=config.model,
                range0=config.range0,
                azimuth=config.azimuth,
                c0=config.c0,
                alpha=config.alpha,
                nugget=config.nugget)

    neigh = Neigh(wradius=config.wradius,
              lookup=config.lookup,
              nb=config.nb)

    params = Params(ndim=config.ndim,
                nx=config.nx,
                ny=config.ny,
                nz=config.nz,
                m=config.m,
                mean=config.mean,
                covar=covar,
                neigh=neigh,
                nnc=config.nnc,
                category=config.category,
                cat_threshold=config.cat_threshold,
                debug=config.debug,
                calc_frob=config.calc_frob,
                seed_search=config.seed_search,
                seed_path=config.seed_path,
                seed_U=config.seed_U
                )

    if params.category:
        params.covar.c0 = params.cat_threshold*(1-params.cat_threshold)


    Rest, means, std, grid, CY, U = sgs_trad(params, params.debug, params.nnc, params.category)
    err = None
    if params.calc_frob:
        err = frob(params, CY, debug=params.debug, nnc=params.nnc)


    if config.cutoff is None:
        config.cutoff = 2*params.covar.range0[0]
    plot_params = Plot(ndim=params.ndim,
                    cutoff=config.cutoff,
                    bins=config.bins,
                    show=config.show,
                    save=config.save,
                    savefilename=config.savefilename,
                    show_NNC=config.show_NNC,
                    category=params.category,
                    cat_threshold=params.cat_threshold,
                    mode=config.mode)

    print(plot_params.dist)

    plot(Rest, means, std, grid, covar, params.m, err, U, plot_params)
