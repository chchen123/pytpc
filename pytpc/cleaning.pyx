# cython: profile=True

import numpy as np
cimport numpy as np
import pandas as pd
from .constants import pi
import math
from scipy.signal import argrelextrema
import cython
from cython.parallel import parallel, prange
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


def linefunc(x, r, t):
    return (r - x * np.cos(t)) / np.sin(t)


def find_peaks(data, pk_width=4):
    maxima = argrelextrema(data, lambda a, b: np.greater_equal(a, b) & np.greater(a, 0),
                           order=2)[0]

    keepmask = data[maxima] != np.roll(data[maxima], -1)
    maxima = maxima[keepmask]

    pkctrs = np.empty_like(maxima, dtype='float64')

    for i, m in enumerate(maxima):
        pkpts = data[m-pk_width:m+pk_width]
        pklocs = np.arange(m-pk_width, m+pk_width)
        pkctrs[i] = (pkpts * pklocs).sum() / pkpts.sum()

    return pkctrs


def clean(xyz, max_distance=40, hough_line_max=2000, hough_line_bins=500, hough_circle_max=500, hough_circle_bins=200):
    xyz = np.ascontiguousarray(xyz)

    # Nearest neighbor cut
    nncts = nearest_neighbor_count(xyz, 10)
    xyz = xyz[nncts > 1]

    # Find center
    cu, cv = hough_circle(xyz, nbins=hough_circle_bins, max_val=hough_circle_max)

    # Find r-theta (arclength)
    rads = np.sqrt((xyz[:, 0] - cu)**2 + (xyz[:, 1] - cv)**2)
    thetas = np.arctan((xyz[:, 1] - cv) / (xyz[:, 0] - cu))
    # xyz['th'] = (xyz.th + pi/2 + 90*degrees) % pi - pi/2
    arclens = rads * thetas

    # Perform linear Hough transform
    linear_hough_data = np.ascontiguousarray(np.column_stack((xyz[:, 2], arclens)))
    lin_space = hough_line(linear_hough_data, nbins=hough_line_bins, max_val=hough_line_max)

    # Find angle of max in Hough space
    maxidx = lin_space.ravel().argsort()  # List of peak indices in increasing order, so look at last few in next step
    thmax = np.floor(np.unravel_index(maxidx[-5:], lin_space.shape)[0].mean()).astype('int')  # Index of average max

    hough_slice = lin_space[thmax - 5:thmax + 5].sum(0)  # Slice of Hough space in angle dimension near max
    hough_slice[hough_slice < 60] = 0

    pkctrs = find_peaks(hough_slice)

    labels = np.full(xyz.shape[0], -1)
    mindists = np.full(xyz.shape[0], np.inf)

    t = thmax * pi / hough_line_bins
    for i, ridx in enumerate(pkctrs.astype('float64')):
        r = ridx * hough_line_max * 2 / hough_line_bins - hough_line_max
        dists = (linefunc(xyz[:, 2], r, t) - arclens)**2
        subset_mask = (dists < mindists) & (dists < max_distance**2)
        labels[subset_mask] = i
        mindists[subset_mask] = dists[subset_mask]

    for label_value in filter(lambda x: x != -1, np.unique(labels)):
        label_set = np.where(labels == label_value)[0]
        if len(label_set) < 10:
            labels[label_set] = -1

    return np.column_stack((xyz, labels, mindists)), (cu, cv)


def find_center_hough(xyzs):
    xyz_order = xyzs[np.argsort(xyzs[:, 2])]

    nbins = 200  # number of bins in r and theta for Hough space discretization
    ths = np.linspace(0, pi, nbins)

    npts = len(xyz_order) - 5
    sqdels = xyz_order[5:, :2]**2 - xyz_order[:-5, :2]**2
    deltas_tiled = np.tile(xyz_order[5:, :2] - xyz_order[:-5, :2], (nbins, 1))

    Hrad = np.empty((nbins * npts, 2))

    Hrad[:, 0] = np.repeat(ths, npts)
    Hrad[:, 1] = (np.tile(np.sum(sqdels, -1), nbins) /
                  (2 * (deltas_tiled[:, 0] * np.cos(Hrad[:, 0])
                        + deltas_tiled[:, 1] * np.sin(Hrad[:, 0]))))

    Hrad = Hrad[1:]

    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:, 0], Hrad[:, 1], nbins, range=((0, pi), (-500, 500)))

    # max bin of histogram
    iRad, jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)
    # convert max bin to theta, radius values
    tRad = iRad * pi / nbins
    rRad = jRad * 1000 / nbins - 500

    # convert theta, r to postions in xy space
    ax = rRad * np.cos(tRad)
    by = rRad * np.sin(tRad)

    return ax, by


def nn_remove_noise(data, radius=40, num_neighbors=10):
    if isinstance(data, pd.DataFrame):
        xyz = data.values
    else:
        xyz = data

    noise = np.sum((np.sqrt(np.sum((xyz[:, None, :3] - xyz[:, :3])**2, -1)) <= radius)
                   & ~np.eye(len(xyz), dtype='bool'), axis=0) < num_neighbors

    if isinstance(data, pd.DataFrame):
        return data.iloc[~noise]
    else:
        return data[~noise]


def apply_clean_cut(raw_xyz_full, qthresh=0.5, nthresh=10, tbthresh=505):
    qcut = raw_xyz_full[:, -1] <= qthresh
    ncut = raw_xyz_full[:, -2] >= nthresh
    tbcut = raw_xyz_full[:, 2] <= tbthresh
    cut = qcut & ncut & tbcut
    return raw_xyz_full[cut, :5]
