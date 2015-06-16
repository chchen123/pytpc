"""
padplane
========

This module contains the functions needed to generate the AT-TPC micromegas pad plane.

To generate a list of points representing the vertices of the triangular micromegas pads, use the function
generate_pad_plane. An angle can be provided to this function if the micromegas is not oriented right-side up.

"""

import math
from math import sin, cos, atan2
import numpy as np
from pytpc.utilities import rot_matrix, skew_matrix
from pytpc.constants import degrees, pi


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

    pads = np.zeros((10240, 3, 2))

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
        rotmat = np.array([[cos(rotation_angle), -sin(rotation_angle)],
                              [sin(rotation_angle), cos(rotation_angle)]])

        # reshape the pads array so we can easily iterate over it
        sh = pads.shape
        pads.shape = (-1, 2)
        for i, v in enumerate(pads):
            pads[i] = np.dot(rotmat, v)
        pads.shape = sh

    return pads


_leftskewmat = skew_matrix(-60.*degrees)
_rightskewmat = skew_matrix(60.*degrees)
_rotneg120mat = rot_matrix(-120*degrees)
_rotpos120mat = rot_matrix(120*degrees)
_rotneg240mat = rot_matrix(-240*degrees)
_rotpos240mat = rot_matrix(240*degrees)
_idmat = np.eye(2)


def _two_pt_line(x, x1, y1, x2, y2):
    """Find the y-value of the equation of the line connecting [x1, y1] to [x2, y2] at the point x.
    """

    return (y2-y1)/(x2-x1) * (x-x1) + y1


def find_pad_coords(coords):
    """Find the center of the nearest pad to the provided x and y coordinates.

    This works as follows:

        1) Divide the pad plane into three 120-degree sectors. Examine the angle that the given point is located at
           and rotate it so that it lies in the first sector. Remember this rotation angle for later.
        2) Apply a skew transformation to the point that would straighten up the triangular pads into rectangular
           (approximately square) pads. Make sure to take into account whether we're in the inner or outer pads.
        3) Using floor division, figure out which rectangular pad the transformed point is in. Each rectangle is
           subdivided into an upper and lower triangle (two triangular pads in each skewed rectangle), so figure out
           which sub-section the point is in by seeing if it's above or below the line dividing the two triangles.
        4) The center of this sub-section is the center of the pad the point is above. Now reverse the skew and rotation
           transformations and return the result.

    Parameters
    ----------
    coords: ndarray
        The [x, y] coordinate pairs for each point

    Returns
    -------
    res : ndarray
        The [x, y] position of the center of the pad the given point is above.

    """

    coords = np.asanyarray(coords)

    # Find the angle to rotate through to bring all points to the first third.
    # Then, rotate the points
    angle = np.arctan2(coords[:, 1], coords[:, 0]) + (np.sign(coords[:, 1])-1)/2*-2*pi
    rotangles = angle // (120*degrees) * (120*degrees) * -1
    rotmats = np.array([[np.cos(rotangles), -np.sin(rotangles)],
                        [np.sin(rotangles),  np.cos(rotangles)]])
    rotmats = np.rollaxis(rotmats, -1)
    rot = np.einsum('ijk,ik->ij', rotmats, coords)  # the rotated points

    # Skew the points to rectify the grid
    rect = np.tensordot(_rightskewmat, rot, ([1], [1])).T

    # Find which points are inside the region of small pads
    isinner = np.all(abs(rect) < [inner_rad * tri_base, inner_rad * tri_height], -1)

    # Get the height and base of the local pads for each point
    t = np.zeros_like(coords)
    t[isinner] = [inner_tri_base, inner_tri_height]
    t[~isinner] = [tri_base, tri_height]

    # Find the first two pad vertices for each point
    v1 = np.floor(rect / t) * t
    v2 = v1 + t

    # The third vertex is either corner of the box. Depends on whether above or below dividing line.
    isabove = rect[:, 1] > _two_pt_line(rect[:, 0], v1[:, 0], v1[:, 1], v2[:, 0], v2[:, 1])
    v3 = np.zeros_like(coords)
    v3[isabove] = v1[isabove] + np.vstack((np.zeros(len(coords)), t[:, 1])).T[isabove]
    v3[~isabove] = v1[~isabove] + np.vstack((t[:, 0], np.zeros(len(isabove)))).T[~isabove]

    # Find the pad centers by averaging, then reskew the grid and rotate back
    res = np.tensordot(_leftskewmat, (v1+v2+v3)/3, ([1], [1])).T
    unrotmats = np.array([[np.cos(-rotangles), -np.sin(-rotangles)],
                          [np.sin(-rotangles),  np.cos(-rotangles)]])
    unrotmats = np.rollaxis(unrotmats, -1)
    rotres = np.einsum('ijk,ik->ij', unrotmats, res)
    return rotres


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

# Make a dictionary of pad centers
keys = tuple(map(tuple, np.round(pads.mean(1)).tolist()))
padcenter_dict = {k: i for i, k in enumerate(keys)}
del keys

del pads