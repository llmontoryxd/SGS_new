from trad import sgs_trad
from models import Params, Covar, Neigh
from plot import plot
from frob import frob

covar = Covar(model='gaussian',
              range0=[10, 10],
              azimuth=[0],
              c0=1.0,
              alpha=1)

neigh = Neigh(wradius=7,
              lookup=True,
              nb=40)

params = Params(nx=65,
                ny=65,
                m=10,
                mean=0.0,
                covar=covar,
                neigh=neigh,
                nnc=False,
                category=False,
                debug=False,
                calc_frob=True
                )

Rest, means, std, grid, CY = sgs_trad(params, debug=params.debug, nnc=params.nnc, category=params.category)
err = None
if params.calc_frob:
    err = frob(params, CY, debug=params.debug, nnc=params.nnc)
plot(Rest, means, std, grid, covar, params.m, err, show=False, save=True, show_NNC=False, mode='vario')
