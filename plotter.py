import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def plot_error(x, y, z, vmin=None, vmax=None, cmap='RdBu',
               title='Localization Error', method='nearest'):
    '''
    Takes x, y, z data and first interpolates the data using the specified
    method and then creates a pseudocolor plot.
    x - x-coordinate of data
    y - y-coordinate of data
    z - value at (x, y) coordinate
    vmin (optional) - value to use for minimum color
                      default: min(z)
    vmax (optional) - value to use for maximum color
                      default: max(z)
    cmap (optional) - color scheme to use for the pseudo colormap
                      default: 'RdBu'
    title (optional) - title to use for the plot
                       default: 'Localization Error'
    method optional - interpolation method to be used
                      default: 'nearest'
                      options: 'nearest', 'linear', 'cubic'
    '''

    if vmin is None:
        vmin = min(z)

    if vmax is None:
        vmax = max(z)

    X, Y = np.meshgrid(x, y)
    Z = interpolate.griddata((x, y), z, (X, Y), method=method)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)

    plt.show()
