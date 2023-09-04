import matplotlib.pyplot as plt
from trad import sgs_trad
from models import Params, Covar, Neigh

covar = Covar(model='exponential',
              range0=[10, 10],
              azimuth=[0],
              c0=1,
              alpha=1)

neigh = Neigh(wradius=3,
              lookup=False,
              nb=40)

params = Params(nx=100,
                ny=50,
                m=1,
                mean=0.0,
                covar=covar,
                neigh=neigh
                )

Rest = sgs_trad(params, debug=False, nnc=True)
c = plt.imshow(Rest)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(c)
plt.show()
