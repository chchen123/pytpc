"""A module for viewing plots of AT-TPC merged data.

This module can be used to generate both two- and three-dimensional plots of data from the AT-TPC. The data should be
imported using the evtdata module. In principle, these functions could also be used to plot arbitrary data.
"""

import math
import numpy
import matplotlib as mpl
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D


def create_triangle(x_offset, y_offset, side, orient):
    """Generates a list of equilateral triangle vertices from the given parameters"""

    return [[x_offset, y_offset],
            [x_offset+side/2, y_offset+orient*side*math.sqrt(3.)/2.],
            [x_offset+side, y_offset]]


def generate_pad_plane():
    """ Generates a list of the vertices of the pads on the pad plane.

    :return: A numpy array of shape (10240, 3, 2)
    """

    small_z_spacing = 2*25.4/1000
    small_tri_side = 184.*25.4/1000
    umega_radius = 10826.772*25.4/1000
    beam_image_radius = 4842.52*25.4/1000
    pad_index = 0

    # small_tri_hi = small_tri_side * math.sqrt(3.)/2.
    small_x_spacing = 2 * small_z_spacing / math.sqrt(3)
    small_y_spacing = small_x_spacing * math.sqrt(3)
    dotted_s_tri_side = 4. * small_x_spacing + small_tri_side
    dotted_s_tri_hi = dotted_s_tri_side * math.sqrt(3) / 2
    dotted_l_tri_side = 2. * dotted_s_tri_side
    dotted_l_tri_hi = dotted_l_tri_side * math.sqrt(3) / 2
    large_x_spacing = small_x_spacing
    large_y_spacing = small_y_spacing
    large_tri_side = dotted_l_tri_side - 4*large_x_spacing
    large_tri_hi = dotted_l_tri_side * math.sqrt(3) / 2
    row_len_s = 2**math.ceil(math.log(beam_image_radius/dotted_s_tri_side) / math.log(2))
    row_len_l = math.floor(umega_radius / dotted_l_tri_hi)
    # hex_side = row_len_s * dotted_s_tri_side

    # Some code was left out here from the original

    xoff = 0
    yoff = 0

    # Create half a circle

    pads = numpy.zeros((10240, 3, 2))

    for j in range(row_len_l):
        pads_in_half_hex = 0
        pads_in_hex = 0
        row_length = abs(math.sqrt(umega_radius**2 - (j*dotted_l_tri_hi + dotted_l_tri_hi/2)**2))

        if j < row_len_s / 2:
            pads_in_half_hex = (2 * row_len_s - 2*j)/4
            pads_in_hex = 2*row_len_s - 1 - 2*j

        pads_in_half_row = row_length / dotted_l_tri_side
        pads_out_half_hex = int(round(2*(pads_in_half_row-pads_in_half_hex)))
        pads_in_row = 2 * pads_out_half_hex + 4 * pads_in_half_hex - 1

        ort = 1

        for i in range(int(pads_in_row)):
            if i == 0:
                if j % 2 == 0:
                    ort = -1
                if ((pads_in_row - 1)/2) % 2 == 1:
                    ort = -ort
            else:
                ort = -ort

            pad_x_off = -(pads_in_half_hex + pads_out_half_hex/2) * dotted_l_tri_side \
                + i * dotted_l_tri_side/2 + 2*large_x_spacing + xoff

            if i < pads_out_half_hex or i > pads_in_hex + pads_out_half_hex - 1 or j > row_len_s / 2 - 1:
                # Outside hex
                pad_y_off = j*dotted_l_tri_hi + large_y_spacing + yoff
                if ort == -1:
                    pad_y_off += large_tri_hi
                pads[pad_index] = create_triangle(pad_x_off, pad_y_off, large_tri_side, ort)
                pad_index += 1
            else:
                # inside hex, make 4 small triangles
                pad_y_off = j*dotted_l_tri_hi + large_y_spacing + yoff

                # Omitted some code here

                if ort == -1:
                    pad_y_off = j*dotted_l_tri_hi + 2*dotted_s_tri_hi - small_y_spacing+yoff

                pads[pad_index] = create_triangle(pad_x_off, pad_y_off, small_tri_side, ort)
                pad_index += 1

                tmp_pad_x_off = pad_x_off + dotted_s_tri_side/2
                tmp_pad_y_off = pad_y_off + ort*dotted_s_tri_hi - 2*ort*small_y_spacing
                pads[pad_index] = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, -ort)
                pad_index += 1

                tmp_pad_y_off = pad_y_off+ort*dotted_s_tri_hi
                pads[pad_index] = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, ort)
                pad_index += 1

                tmp_pad_x_off = pad_x_off + dotted_s_tri_side
                pads[pad_index] = create_triangle(tmp_pad_x_off, pad_y_off, small_tri_side, ort)
                pad_index += 1

    # Create symmetric pads
    for i in range(pad_index):
        pads[pad_index + i] = pads[i] * [1, -1]

    return pads


def show_pad_plane():
    """Displays the pad plane"""

    pts = generate_pad_plane()
    c = mpl.collections.PolyCollection(pts)

    fig, ax = plt.subplots()

    ax.add_collection(c)
    ax.autoscale_view()

    fig.show()


def _make_pad_colormap():
    """Generates the color map used for the pad plots.

    This differs from the standard colormap in that zero is mapped to gray.
    """

    carr = mpl.cm.ScalarMappable(cmap='rainbow').to_rgba(range(256))[:, :-1]
    carr[0] = numpy.array([230./256., 230./256., 230./256.])
    return mpl.colors.LinearSegmentedColormap.from_list('mymap', carr, N=256)


def pad_plot(data):
    """ Plot the given data on the pads of the Micromegas.

    :param data: The data, as an array indexed by pad number.
    """

    mycm = _make_pad_colormap()
    sm = mpl.cm.ScalarMappable(cmap=mycm)
    sm.set_array(data)
    colors = sm.to_rgba(data)

    pts = generate_pad_plane()
    c = mpl.collections.PolyCollection(pts, facecolors=colors, edgecolors='white', linewidths=0.05)

    fig, ax = plt.subplots()

    ax.add_collection(c)

    plt.axis('equal')
    plt.xlabel('X Position [mm]')
    plt.ylabel('Y Position [mm]')

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