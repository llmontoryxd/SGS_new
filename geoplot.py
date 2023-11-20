from pylab import *
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import variograms, utilities

cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.5, 225 / 255., 225 / 255.),
                 (0.75, 0.141, 0.141),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.5, 57 / 255., 57 / 255.),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.376, 0.376),
                  (0.5, 198 / 255., 198 / 255.),
                  (0.75, 1.0, 1.0),
                  (1.0, 0.0, 0.0))}

YPcmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)