import scipy
import scipy.stats
import numpy as np
from scipy.spatial.distance import pdist, squareform


def pairwise( data ):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
    '''
    # determine the size of the data
    npoints, cols = data.shape
    # give a warning for large data sets
    if npoints > 10000:
        print("You have more than 10,000 data points, this might take a minute.")
    # return the square distance matrix
    return squareform( pdist( data[:,:2] ) )

def readGeoEAS( fn ):
    '''
    Input:  (fn)   filename describing a GeoEAS file
    Output: (data) NumPy array
    --------------------------------------------------
    Read GeoEAS files as described by the GSLIB manual
    '''
    f = open( fn, "r" )
    # title of the data set
    title = f.readline()
    # number of variables
    nvar = int( f.readline() )
    # variable names
    columns = [ f.readline().strip() for i in range( nvar ) ]
    # make a list for the data
    data = list()
    # for each line of the data
    while True:
        # read a line
        line = f.readline()
        # if that line is empty
        if line == '':
            # the jump out of the while loop
            break
        # otherwise, append that line to the data list
        else:
            data.append( line )
    # strip off the newlines, and split by whitespace
    data = [ i.strip().split() for i in data ]
    # turn a list of list of strings into an array of floats
    data = np.array( data, dtype=np.double )
    # combine the data with the variable names into a DataFrame
    # df = pandas.DataFrame( data, columns=columns,  )
    return data
