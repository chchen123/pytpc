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
from pytpc.utilities import numpyize


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


def pad_plot(data, pads=None, scale='log', cmap=pad_cm, cmin=None, cmax=None):
    """ Plot the given data on the pads of the Micromegas.

    Parameters
    ----------
    data : array-like
        The data to be plotted. This should be a 1-D array with an entry for each pad.
    pads : array-like, optional
        The vertices of the pads. If this is not provided, the default pad plane will be generated using
        `generate_pad_plane`. This can be used for a rotated pad plane, for example.
    scale : string, optional
        Scale for color map. 'log' for log scale, or 'linear' for linear.
    cmap : string or Matplotlib color map, optional
        Matplotlib color map or name of a color map
    cmin, cmax: float, optional
        The min and max values for the color scale. If omitted, Matplotlib will choose its default values.

    Returns
    -------
    fig : Figure
        The generated figure
    sm : matplotlib.cm.ScalarMappable
        The color mapping object used in the plot. This can be used to generate a colorbar with
        `plt.colorbar(sm)`.
    """

    data = numpy.asanyarray(data)

    if scale is 'log':
        nm = LogNorm()
    elif scale is 'linear':
        nm = None
    else:
        raise ValueError('invalid scale. Must be in set {}'.format(('log', 'linear')))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=nm)
    sm.set_array(data)
    sm.set_clim(cmin, cmax)
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

    return fig, sm


def chamber_plot(data, hits=None, pads=None, zscale='time'):
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
    zscale : string, optional
        The type of data to be plotted on the z axis. Options are ['time', 'dist'].

    Returns
    -------
    fig : matplotlib figure
        The generated figure
    """

    data = numpy.asanyarray(data)

    if zscale is 'time':
        zmax = 512
    elif zscale is 'dist':
        zmax = 1000  # FIXME: Change these units to be consistent with simulation (m)
    else:
        raise ValueError('invalid zscale: {}'.format(zscale))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_bgcolor('none')
    ax.view_init(azim=0, elev=15)
    ax.set_xlim(-250, 250)
    ax.set_ylim(-250, 250)
    ax.set_zlim(0, zmax)
    sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', linewidth=0, s=2, cmap=ch_cm,
                    norm=LogNorm(), c=data[:, 3])

    # Draw the anode and cathode at the ends
    anode = mpl.patches.CirclePolygon((0, 0), radius=250, resolution=50)
    cathode = mpl.patches.CirclePolygon((0, 0), radius=250, resolution=50)
    ends = mpl.collections.PolyCollection([anode.get_verts(), cathode.get_verts()],
                                          facecolors=['#c4cccc', 'none'], edgecolors=['none', 'black'])
    ax.add_collection3d(ends, zs=[0, zmax-1])

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


@numpyize
def state_vector_plots(x_act=None, act=None, x_calc=None, calc=None, x_data=None, data=None, covar=None):
    """Plot the state vectors and compare them to the data.

    This would mainly be used to see the results of the fit.

    Each set of parameters is optional, but if one y parameter is included, its corresponding x parameter must also
    be present. For instance, if `calc` is provided, then `x_calc` must also be provided.

    Parameters
    ----------
    x_act : array-like, optional
        x values of the actual state vector
    act : array-like, optional
        actual state vector
    x_calc : array-like, optional
        x values of calculated state vector
    calc : array-like, optional
        the calculated state vector
    x_data : array-like, optional
        x values of the data
    data : array-like, optional
        measured data points
    covar : array-like, optional
        the covariance matrix of the fit

    Returns
    -------
    figure
        The state vector plots
    """
    fig, ax = plt.subplots(3, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    blue = sns.xkcd_rgb['denim blue']
    red = sns.xkcd_rgb['crimson']
    purple = sns.xkcd_rgb['amethyst']

    if x_act is not None and act is not None:
        ax[0, 0].plot(x_act, act[:, 0], color=blue, label='Actual', zorder=2)
        ax[1, 0].plot(x_act, act[:, 1], color=blue, label='Actual', zorder=2)
        ax[2, 0].plot(x_act, act[:, 2], color=blue, label='Actual', zorder=2)
        ax[0, 1].plot(x_act, act[:, 3], color=blue, label='Actual', zorder=2)
        ax[1, 1].plot(x_act, act[:, 4], color=blue, label='Actual', zorder=2)
        ax[2, 1].plot(x_act, act[:, 5], color=blue, label='Actual', zorder=2)

    if calc is not None and x_calc is not None:
        ax[0, 0].plot(x_calc, calc[:, 0], color=purple, label='Calculated', zorder=3)
        ax[1, 0].plot(x_calc, calc[:, 1], color=purple, label='Calculated', zorder=3)
        ax[2, 0].plot(x_calc, calc[:, 2], color=purple, label='Calculated', zorder=3)
        ax[0, 1].plot(x_calc, calc[:, 3], color=purple, label='Calculated', zorder=3)
        ax[1, 1].plot(x_calc, calc[:, 4], color=purple, label='Calculated', zorder=3)
        ax[2, 1].plot(x_calc, calc[:, 5], color=purple, label='Calculated', zorder=3)

    if data is not None and x_data is not None:
        ax[0, 0].plot(x_data, data[:, 0], '.', markersize=4, color=red, label='Data', zorder=1, alpha=0.5)
        ax[1, 0].plot(x_data, data[:, 1], '.', markersize=4, color=red, label='Data', zorder=1, alpha=0.5)
        ax[2, 0].plot(x_data, data[:, 2], '.', markersize=4, color=red, label='Data', zorder=1, alpha=0.5)

    if covar is not None:
        ubd = calc + numpy.sqrt(numpy.diagonal(covar, axis1=1, axis2=2))
        lbd = calc - numpy.sqrt(numpy.diagonal(covar, axis1=1, axis2=2))
        ax[0, 0].fill_between(x_calc, ubd[:, 0], lbd[:, 0], color=purple, alpha=0.3, zorder=0)
        ax[1, 0].fill_between(x_calc, ubd[:, 1], lbd[:, 1], color=purple, alpha=0.3, zorder=0)
        ax[2, 0].fill_between(x_calc, ubd[:, 2], lbd[:, 2], color=purple, alpha=0.3, zorder=0)
        ax[0, 1].fill_between(x_calc, ubd[:, 3], lbd[:, 3], color=purple, alpha=0.3, zorder=0)
        ax[1, 1].fill_between(x_calc, ubd[:, 4], lbd[:, 4], color=purple, alpha=0.3, zorder=0)
        ax[2, 1].fill_between(x_calc, ubd[:, 5], lbd[:, 5], color=purple, alpha=0.3, zorder=0)

    ax[0, 0].set_ylabel('x [m]')
    ax[0, 0].legend(loc='best')

    ax[1, 0].set_ylabel('y [m]')
    ax[1, 0].legend(loc='best')

    ax[2, 0].set_ylabel('z [m]')
    ax[2, 0].legend(loc='best')

    ax[0, 1].set_ylabel('px [MeV/c]')
    ax[0, 1].legend(loc='best')

    ax[1, 1].set_ylabel('py [MeV/c]')
    ax[1, 1].legend(loc='best')

    ax[2, 1].set_ylabel('pz [MeV/c]')
    ax[2, 1].legend(loc='best')

    return fig
