from trad import sgs_trad
from models import Params, Covar, Neigh, Config, Plot
from plot import plot
from frob import frob


def calculate(configfilename):
    config = Config(configfilename)

    covar = Covar(model=config.model,
                range0=config.range0,
                azimuth=config.azimuth,
              c0=config.c0,
              alpha=config.alpha)

    neigh = Neigh(wradius=config.wradius,
              lookup=config.lookup,
              nb=config.nb)

    params = Params(nx=config.nx,
                ny=config.ny,
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

    Rest, means, std, grid, CY, U = sgs_trad(params, debug=params.debug, nnc=params.nnc, category=params.category)
    err = None
    if params.calc_frob:
        err = frob(params, CY, debug=params.debug, nnc=params.nnc)

    plot_params = Plot(cutoff=config.cutoff,
                   bins=config.bins,
                   show=config.show,
                   save=config.save,
                   savefilename=config.savefilename,
                   show_NNC=config.show_NNC,
                   mode=config.mode)
    plot(Rest, means, std, grid, covar, params.m, err, U, plot_params)
