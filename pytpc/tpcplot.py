"""A module for viewing plots of AT-TPC merged data.

This module can be used to generate both two- and three-dimensional plots of data from the AT-TPC. The data should be
imported using the evtdata module. In principle, these functions could also be used to plot arbitrary data.
"""

from __future__ import division, print_function
import numpy
import matplotlib as mpl
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D

from pytpc.padplane import generate_pad_plane


def show_pad_plane(pads=None):
    """Displays the pad plane"""

    if pads is None:
        pads = generate_pad_plane()

    c = mpl.collections.PolyCollection(pads)

    fig, ax = plt.subplots()

    ax.add_collection(c)
    ax.autoscale_view()

    fig.show()


def _make_pad_colormap():
    """Generates the color map used for the pad plots.

    This differs from the standard colormap in that zero is mapped to gray.
    """

    carr = mpl.cm.ScalarMappable(cmap='YlGnBu').to_rgba(range(256))[:, :-1]
    carr[0] = numpy.array([230./256., 230./256., 230./256.])
    return mpl.colors.LinearSegmentedColormap.from_list('mymap', carr, N=256)


def pad_plot(data, pads=None):
    """ Plot the given data on the pads of the Micromegas.

    :param data: The data, as an array indexed by pad number.
    """

    mycm = _make_pad_colormap()
    sm = mpl.cm.ScalarMappable(cmap='GnBu', norm=matplotlib.colors.LogNorm())
    sm.set_array(data)
    colors = sm.to_rgba(data)

    if pads is None:
        pads = generate_pad_plane()

    c = mpl.collections.PolyCollection(pads, facecolors=colors, edgecolors='none')
    cbg = mpl.collections.PolyCollection(pads, facecolors='white', edgecolors='none')

    c.set_zorder(2)
    cbg.set_zorder(1)

    fig, ax = plt.subplots()

    ax.axison = False

    bdcirc = plt.Circle((0, 0), radius=290., facecolor='#c4cccc', edgecolor='none')
    bdcirc.set_zorder(0)
    ax.add_artist(bdcirc)

    ax.add_collection(cbg)
    ax.add_collection(c)

    plt.axis('equal')

    cbar = fig.colorbar(sm)
    cbar.set_label('Activation')

    return fig


def chamber_plot(data):
    """ Plot the given data in 3D.

    The data should be four-dimensional. The first three dimensions are the coordinates of the data points, and the
    last dimension should contain the weight (or energy / activation) of the point. This is used for coloring.

    :param data: The data, as an array of [x, y, z, weight]
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker=',', linewidth=0, s=1, cmap='rainbow', c=data[:, 3])

    return fig