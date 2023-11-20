import utilities, variograms, model, k3d, geoplot
import matplotlib.pyplot as plt
import numpy as np
import pandas

z = utilities.readGeoEAS('data/zoneA.txt')
P = z[:, [0, 1, 3]]

pt = [2000, 4700]



plt.scatter(P[:,0], P[:,1], c=P[:,2], cmap=geoplot.YPcmap)
plt.title('Zone A Subset % Porosity')
plt.colorbar()
xmin, xmax = 0, 4250
ymin, ymax = 3200, 6250
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
for i in range(len(P[:,2])):
    x, y, por = P[i]
    if (x < xmax) & (y > ymin) & (y < ymax):
        plt.text( x+100, y, '{:4.2f}'.format( por ) )
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)');

tolerance = 250
lags = np.arange(tolerance, 10000, tolerance*2)
sill = np.var(P[:,2])
svm = model.semivariance(model.spherical, (4000, sill))


covfct = model.covariance(model.spherical, (4000, sill))



est, kstd = k3d.krige(P, covfct, [[2000,4700],[2100,4700],[2000,4800],[2100,4800]], 'simple', N=6)
plt.scatter(2000, 4700, c=est[0], cmap=geoplot.YPcmap)

plt.show()



