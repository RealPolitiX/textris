import numpy as np
import matplotlib.colors as mplc


class MidpointNormalize(mplc.Normalize):
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Ignoring masked values
        
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

def gini(array):
    '''Gini coefficient of a numpy array,
    adapted from Olivia Guest'''

    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    
    array = np.sort(array)
    index = np.arange(1, array.shape[0]+1)
    n = array.shape[0]
    
    # Calculate the Gini coefficient
    gnco = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    
    return gnco


def hoyer_squared(array):
    # Squared Hoyer sparsity measure
    
    return np.abs(array).sum()**2 / np.sum(array**2)


def hoyer(array):
    # Original Hoyer sparsity measure
    
    n = len(array)
    ratio = np.abs(array).sum() / np.sqrt((array**2).sum())
    return (np.sqrt(n) - ratio) / (np.sqrt(n) - 1)