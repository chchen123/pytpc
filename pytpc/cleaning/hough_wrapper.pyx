"""hough_wrapper.pyx

A Cython wrapper around `hough.c` that provides the Hough transform functions.

"""

import numpy as np
cimport numpy as np
import cython
from libc.stdint cimport int64_t
from libc.math cimport sin, cos, M_PI


cdef extern from "hough.h":
    void houghline(const double *xy, const int dim0, const int dim1, int64_t *accum, const int nbins, const double max_val)
    void houghcircle(const double *xy, const int dim0, const int dim1, int64_t *accum, const int nbins, const double max_val)
    void neighborcount(const double *xy, const int nrows, const int ncols, int64_t *counts, const double radius)


def hough_line(np.ndarray[np.double_t, ndim=2, mode='c'] xy, int nbins=500, int max_val=2000):
    """Performs the linear Hough transform and returns the Hough space.

    The generated Hough space has the angle along dimension 0 and the radius along dimension 1. The radii range
    from `-max_val` to `max_val`. The angles range from 0 to pi.

    Parameters
    ----------
    xy : ndarray
        An array with the positional data to transform. The dimensions should be (N, 2), and the data must
        have C-style row-major ordering.
    nbins : int, optional
        The number of bins to use in each dimension of the Hough space accumulator. The default value is 500.
    max_val : int, optional
        The extreme value in the Hough radius dimension. The default value is 2000.

    Returns
    -------
    accum : ndarray
        The Hough space accumulator, a 2D array with dimension `(nbins, nbins)`.

    """
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] accum = np.zeros((nbins, nbins), dtype=np.int64)
    cdef double* xyPtr = <double*> xy.data
    cdef int64_t* accumPtr = <int64_t*> accum.data
    cdef int dim0 = xy.shape[0]
    cdef int dim1 = xy.shape[1]

    houghline(xyPtr, dim0, dim1, accumPtr, nbins, max_val)

    return accum


def hough_circle(np.ndarray[np.double_t, ndim=2, mode='c'] xy, int nbins=200, int max_val=500):
    """Performs the Hough transform for circles to find the center of a spiral.

    Parameters
    ----------
    xy : ndarray
        A 2D array of data with dimension `(N, 2)` and C-style row-major ordering.
    nbins : int, optional
        The number of bins to use in each dimension of the Hough space accumulator. The default value is 200.
    max_val : int, optional
        The extreme value in the Hough radius dimension. The default value is 500.

    Returns
    -------
    cx, cy : float
        The center of the spiral.

    """
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] accum = np.zeros((nbins, nbins), dtype=np.int64)
    cdef double* xyPtr = <double*> xy.data
    cdef int64_t* accumPtr = <int64_t*> accum.data
    cdef int dim0 = xy.shape[0]
    cdef int dim1 = xy.shape[1]

    houghcircle(xyPtr, dim0, dim1, accumPtr, nbins, max_val)

    cdef int imax, jmax
    imax, jmax = np.unravel_index(accum.argmax(), (nbins, nbins))
    # convert max bin to theta, radius values
    cdef double tRad = imax * M_PI / nbins
    cdef double rRad = jmax * 2 * max_val / nbins - max_val

    # convert theta, r to postions in xy space
    cdef double cx = rRad * cos(tRad)
    cdef double cy = rRad * sin(tRad)

    return cx, cy


def nearest_neighbor_count(np.ndarray[np.double_t, ndim=2, mode='c'] xyz, double radius):
    """Count the number of neighbors each point has within a given radius.

    Parameters
    ----------
    xyz : ndarray
        An Nx3 set of (x, y, z) data with C-style row-major ordering.
    radius : float
        The neighborhood radius.

    Returns
    -------
    counts : ndarray
        A one-dimensional array of length N containing the number of neighbors for each point in `xyz`.

    """
    cdef np.ndarray[np.int64_t, ndim=1] counts = np.zeros(xyz.shape[0], dtype=np.int64)

    cdef double* xyzPtr = <double*> xyz.data
    cdef int64_t* countsPtr = <int64_t*> counts.data

    neighborcount(xyzPtr, xyz.shape[0], xyz.shape[1], countsPtr, radius)
    return counts
