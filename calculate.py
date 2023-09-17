from trad import sgs_trad
from models import Params, Covar, Neigh
from plot import plot

covar = Covar(model='gaussian',
              range0=[10, 10],
              azimuth=[0],
              c0=0.1,
              alpha=1)

neigh = Neigh(wradius=3,
              lookup=True,
              nb=40)

params = Params(nx=50,
                ny=50,
                m=1,
                mean=0.0,
                covar=covar,
                neigh=neigh
                )

Rest, means, std, grid = sgs_trad(params, debug=False, nnc=True, category=False)
plot(Rest, means, std, grid, covar, params.m, show=False, save=True, show_NNC=False, mode='vario')
