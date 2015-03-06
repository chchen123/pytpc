"""
padplane
========

This module contains the functions needed to generate the AT-TPC micromegas pad plane.

To generate a list of points representing the vertices of the triangular micromegas pads, use the function
generate_pad_plane. An angle can be provided to this function if the micromegas is not oriented right-side up.

"""

import math
from math import sin, cos
import numpy


def create_triangle(x_offset, y_offset, side, orient):
    """Generates a list of equilateral triangle vertices from the given parameters"""

    return [[x_offset, y_offset],
            [x_offset + side / 2, y_offset + orient * side * math.sqrt(3.) / 2.],
            [x_offset + side, y_offset]]


def generate_pad_plane(rotation_angle=None):
    """ Generates a list of the vertices of the pads on the pad plane.

    The pads themselves are (approximately) equilateral triangles. The first and last vertices
    represent the ends of the base of the triangle, and the middle vertex is the tip. The points
    are given as [x, y] pairs. The pads are listed in order of increasing pad number.

    Parameters
    ----------
    rotation_angle : int or float
        The angle (in rad) by which to rotate the result. This is useful if the micromegas is not right-side-up.

    Returns
    -------
    pads : ndarray
        A list of vertices of the pads.
    """

    small_z_spacing = 2 * 25.4 / 1000
    small_tri_side = 184. * 25.4 / 1000
    umega_radius = 10826.772 * 25.4 / 1000
    beam_image_radius = 4842.52 * 25.4 / 1000
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
    large_tri_side = dotted_l_tri_side - 4 * large_x_spacing
    large_tri_hi = dotted_l_tri_side * math.sqrt(3) / 2
    row_len_s = 2 ** math.ceil(math.log(beam_image_radius / dotted_s_tri_side) / math.log(2))
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
        row_length = abs(math.sqrt(umega_radius ** 2 - (j * dotted_l_tri_hi + dotted_l_tri_hi / 2) ** 2))

        if j < row_len_s / 2:
            pads_in_half_hex = (2 * row_len_s - 2 * j) / 4
            pads_in_hex = 2 * row_len_s - 1 - 2 * j

        pads_in_half_row = row_length / dotted_l_tri_side
        pads_out_half_hex = int(round(2 * (pads_in_half_row - pads_in_half_hex)))
        pads_in_row = 2 * pads_out_half_hex + 4 * pads_in_half_hex - 1

        ort = 1

        for i in range(int(pads_in_row)):
            if i == 0:
                if j % 2 == 0:
                    ort = -1
                if ((pads_in_row - 1) / 2) % 2 == 1:
                    ort = -ort
            else:
                ort = -ort

            pad_x_off = -(pads_in_half_hex + pads_out_half_hex / 2) * dotted_l_tri_side \
                        + i * dotted_l_tri_side / 2 + 2 * large_x_spacing + xoff

            if i < pads_out_half_hex or i > pads_in_hex + pads_out_half_hex - 1 or j > row_len_s / 2 - 1:
                # Outside hex
                pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff
                if ort == -1:
                    pad_y_off += large_tri_hi
                pads[pad_index] = create_triangle(pad_x_off, pad_y_off, large_tri_side, ort)
                pad_index += 1
            else:
                # inside hex, make 4 small triangles
                pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff

                # Omitted some code here

                if ort == -1:
                    pad_y_off = j * dotted_l_tri_hi + 2 * dotted_s_tri_hi - small_y_spacing + yoff

                pads[pad_index] = create_triangle(pad_x_off, pad_y_off, small_tri_side, ort)
                pad_index += 1

                tmp_pad_x_off = pad_x_off + dotted_s_tri_side / 2
                tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi - 2 * ort * small_y_spacing
                pads[pad_index] = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, -ort)
                pad_index += 1

                tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi
                pads[pad_index] = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, ort)
                pad_index += 1

                tmp_pad_x_off = pad_x_off + dotted_s_tri_side
                pads[pad_index] = create_triangle(tmp_pad_x_off, pad_y_off, small_tri_side, ort)
                pad_index += 1

    # Create symmetric pads
    for i in range(pad_index):
        pads[pad_index + i] = pads[i] * [1, -1]

    if rotation_angle is not None:
        rotmat = numpy.array([[cos(rotation_angle), -sin(rotation_angle)],
                              [sin(rotation_angle), cos(rotation_angle)]])

        # reshape the pads array so we can easily iterate over it
        sh = pads.shape
        pads.shape = (-1, 2)
        for i, v in enumerate(pads):
            pads[i] = numpy.dot(rotmat, v)
        pads.shape = sh

    return pads


# Now generate some parameters about the pads

pads = generate_pad_plane()

pad_height = pads[0, 1, 1] - pads[0, 0, 1]  #: The height of an outer pad
inner_pad_height = pads[50, 1, 1] - pads[50, 0, 1]  #: The height of an inner pad
pad_base = pads[0, 2, 0] - pads[0, 0, 0]  #: The base of an outer pad
inner_pad_base = pads[50, 2, 0] - pads[50, 0, 0]  #: The base of an inner pad
xgap = pads[153, 0, 0] - pads[152, 1, 0]  #: The horizontal gap between adjacent pads
ygap = pads[152, 0, 1] - pads[5272, 0, 1]  #: The vertical gap between adjacent pads

tri_height = pad_height + ygap  #: The height of an outer pad, including the gap
tri_base = pad_base + 2. * xgap  #: The base of an outer pad, including the gap
inner_tri_base = tri_base / 2.  #: The base of an inner pad, including the gap
inner_tri_height = tri_height / 2.  #: The height of an inner pad, including the gap

inner_rad = 16

del pads