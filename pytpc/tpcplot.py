"""A module for viewing plots of AT-TPC merged data.

This module can be used to generate both two- and three-dimensional plots of data from the AT-TPC. The data should be
imported using the evtdata module. In principle, these functions could also be used to plot arbitrary data.
"""

from __future__ import division, print_function
import numpy
import matplotlib as mpl
import matplotlib.collections
from matplotlib.patches import Circle, PathPatch
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LogNorm
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import seaborn as sns

from pytpc.padplane import generate_pad_plane


# Define colormaps for the plots
pad_colors = sns.cubehelix_palette(n_colors=6, start=0, rot=-0.4,
                                   gamma=1, hue=0.8, light=0.95, dark=0.15)
pad_cm = sns.blend_palette(pad_colors, as_cmap=True)
"""The color map for the `pad_plot` function."""

ch_colors = sns.cubehelix_palette(n_colors=6, start=0, rot=-0.4,
                                  gamma=1, hue=1, light=0.75, dark=0.1)
ch_cm = sns.blend_palette(ch_colors, as_cmap=True)
"""The color map for the `chamber_plot` function."""


def show_pad_plane(pads=None):
    """Displays the pad plane"""

    if pads is None:
        pads = generate_pad_plane()

    c = mpl.collections.PolyCollection(pads)

    fig, ax = plt.subplots()

    ax.add_collection(c)
    ax.autoscale_view()

    fig.show()


def _generate_pad_collection(data, pads=None):

    sm = mpl.cm.ScalarMappable(cmap=pad_cm, norm=LogNorm())
    sm.set_array(data)
    colors = sm.to_rgba(data)

    if pads is None:
        pads = generate_pad_plane()

    c = mpl.collections.PolyCollection(pads, facecolors=colors, edgecolors='none')
    return c


def pad_plot(data, pads=None):
    """ Plot the given data on the pads of the Micromegas.

    Parameters
    ----------
    data : array-like
        The data to be plotted. This should be a 1-D array with an entry for each pad.
    pads : array-like, optional
        The vertices of the pads. If this is not provided, the default pad plane will be generated using
        `generate_pad_plane`. This can be used for a rotated pad plane, for example.

    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """

    data = numpy.asanyarray(data)

    sm = mpl.cm.ScalarMappable(cmap=pad_cm, norm=LogNorm())
    sm.set_array(data)
    colors = sm.to_rgba(data)

    if pads is None:
        pads = generate_pad_plane()
    else:
        pads = numpy.asanyarray(pads)

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


def chamber_plot(data, hits=None, pads=None):
    """ Plot the given data in 3D.

    The data should be four-dimensional. The first three dimensions are the coordinates of the data points, and the
    last dimension should contain the weight (or energy / activation) of the point. This is used for coloring.

    Parameters
    ----------
    data : array-like
        The data to be plotted. The format should be as described above.
    hits : array-like, optional
        The hits on the pad plane, as a 1-D array with an entry for each pad. If provided, the hit pads will be
        drawn on the cathode.
    pads : array-like, optional
        The vertices of the pads. If this is not provided, the default pad plane will be generated using
        `generate_pad_plane`. This can be used for a rotated pad plane, for example.

    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """

    data = numpy.asanyarray(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_bgcolor('none')
    ax.view_init(azim=0, elev=15)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(0, 512)
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', linewidth=0, s=2, cmap=ch_cm,
                    norm=LogNorm(), c=data[:, 3])

    # Draw the anode and cathode at the ends
    anode = mpl.patches.CirclePolygon((0, 0), radius=250, resolution=50)
    cathode = mpl.patches.CirclePolygon((0, 0), radius=250, resolution=50)
    ends = mpl.collections.PolyCollection([anode.get_verts(), cathode.get_verts()],
                                          facecolors=['#c4cccc', 'none'], edgecolors=['none', 'black'])
    ax.add_collection3d(ends, zs=[0, 511])

    if hits is not None:
        pads = _generate_pad_collection(hits, pads)
        ax.add_collection3d(pads, zs=0)

    # Add a color scale
    cbar = fig.colorbar(sc, shrink=0.75)
    cbar.set_label('Activation')

    return fig


def event_view(evt, pads=None):

    if pads is None:
        pads = generate_pad_plane()

    data = evt.xyzs(pads)
    hits = evt.hits()
    mesh = evt.traces['data'].sum(0)

    fig = plt.figure()

    chax = fig.add_subplot(223)
    chax.set_xlim(0, 511)
    chax.set_ylim(-290, 290)
    chax.set_axis_bgcolor('#c4cccc')
    chax.scatter(data[:, 2], data[:, 1], marker='.', linewidths=0, s=2, cmap='winter', norm=LogNorm(), c=data[:, 3])
    for sp in chax.spines.values():
        sp.visible = False

    msax = fig.add_subplot(221, sharex=chax)
    msax.plot(mesh)

    pdax = fig.add_subplot(224, sharey=chax)
    pdax.axison = False
    pdax.set_aspect('equal')
    pdax.set_xlim(-250, 250)

    hit_pads = _generate_pad_collection(hits, pads)
    bg_pads = mpl.collections.PolyCollection(pads, facecolors='white', edgecolors='none')
    bg_circ = plt.Circle((0, 0), radius=290., facecolor='#c4cccc', edgecolor='none')

    bg_circ.set_zorder(0)
    bg_pads.set_zorder(1)
    hit_pads.set_zorder(2)

    pdax.add_artist(bg_circ)
    pdax.add_collection(bg_pads)
    pdax.add_collection(hit_pads)


    return fig