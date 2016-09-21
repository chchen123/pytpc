# cython: profile=True

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
    """ Finds lines in data

    """
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] accum = np.zeros((nbins, nbins), dtype=np.int64)
    cdef double* xyPtr = <double*> xy.data
    cdef int64_t* accumPtr = <int64_t*> accum.data
    cdef int dim0 = xy.shape[0]
    cdef int dim1 = xy.shape[1]

    houghline(xyPtr, dim0, dim1, accumPtr, nbins, max_val)

    return accum


def hough_circle(np.ndarray[np.double_t, ndim=2, mode='c'] xy, int nbins=200, int max_val=500):
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
    cdef np.ndarray[np.int64_t, ndim=1] counts = np.zeros(xyz.shape[0], dtype=np.int64)

    cdef double* xyzPtr = <double*> xyz.data
    cdef int64_t* countsPtr = <int64_t*> counts.data

    neighborcount(xyzPtr, xyz.shape[0], xyz.shape[1], countsPtr, radius)
    return counts
