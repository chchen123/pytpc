import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from .hough_wrapper import hough_line, hough_circle, nearest_neighbor_count
from .constants import pi


def linefunc(x, r, t):
    """Function of a line in the linear Hough space."""
    return (r - x * np.cos(t)) / np.sin(t)


class HoughCleaner(object):
    def __init__(self, config):
        self.peak_width = config['peak_width']

        self.linear_hough_max = config['hough_line_max']
        self.linear_hough_nbins = config['linear_hough_nbins']
        self.circle_hough_max = config['circle_hough_max']
        self.circle_hough_nbins = config['circle_hough_nbins']

        self.max_distance_from_line = config['clean_max_distance_from_line']
        self.min_pts_per_line = config['clean_min_pts_per_line']
        self.min_num_neighbors = config['clean_min_num_neighbors']
        self.neighbor_radius = config['clean_neighbor_radius']

    def find_peaks(self, data):

        def comparator(a, b):
            return np.greater_equal(a, b) & np.greater(a, 0)

        maxima = argrelextrema(data, comparator, order=2)[0]

        keepmask = data[maxima] != np.roll(data[maxima], -1)
        maxima = maxima[keepmask]

        pkctrs = np.empty_like(maxima, dtype='float64')

        for i, m in enumerate(maxima):
            first = m - self.peak_width
            last = m + self.peak_width
            pkpts = data[first:last]
            pklocs = np.arange(first, last)
            pkctrs[i] = (pkpts * pklocs).sum() / pkpts.sum()

        return pkctrs

    def linhough_theta_from_bin(self, t):
        return t * pi / self.linear_hough_nbins

    def linhough_rad_from_bin(self, r):
        return r * self.linear_hough_max * 2 / self.linear_hough_nbins - self.linear_hough_max

    def classify_points(self, xyz, arclens, theta, radii):
        labels = np.full(xyz.shape[0], -1, dtype='int64')          # Group labels for each xyz point
        mindists = np.full(xyz.shape[0], np.inf, dtype='float64')  # Distance to closest line for each xyz point

        # Find min distances for each identified line
        for i, rad in enumerate(radii):
            dists = (linefunc(xyz[:, 2], rad, theta) - arclens)**2
            subset_mask = (dists < mindists) & (dists < self.max_distance_from_line**2)
            labels[subset_mask] = i
            mindists[subset_mask] = dists[subset_mask]

        for label_value in filter(lambda x: x != -1, np.unique(labels)):
            label_set = np.where(labels == label_value)[0]
            if len(label_set) < self.min_pts_per_line:
                labels[label_set] = -1

        return (labels, mindists)

    def clean(self, xyz):
        xyz = np.ascontiguousarray(xyz)

        # Nearest neighbor cut
        nncts = nearest_neighbor_count(xyz, self.neighbor_radius)
        xyz = xyz[nncts > self.min_num_neighbors]

        # Find center
        cu, cv = hough_circle(xyz, nbins=self.hough_circle_bins, max_val=self.hough_circle_max)

        # Find r-theta (arclength)
        rads = np.sqrt((xyz[:, 0] - cu)**2 + (xyz[:, 1] - cv)**2)
        thetas = np.arctan((xyz[:, 1] - cv) / (xyz[:, 0] - cu))
        # xyz['th'] = (xyz.th + pi/2 + 90*degrees) % pi - pi/2
        arclens = rads * thetas

        # Perform linear Hough transform
        linear_hough_data = np.ascontiguousarray(np.column_stack((xyz[:, 2], arclens)))
        lin_space = hough_line(linear_hough_data, nbins=self.hough_line_bins, max_val=self.hough_line_max)

        # Find angle of max in Hough space
        maxidx = lin_space.ravel().argsort()  # List of peak indices in increasing order: look at last few in next step
        thmax = np.floor(np.unravel_index(maxidx[-5:], lin_space.shape)[0].mean()).astype('int')  # Index of average max

        # Slice Hough space in angle dimension near max
        hough_slice = lin_space[thmax - 5:thmax + 5].sum(0)
        hough_slice[hough_slice < 60] = 0  # Use a threshold to get rid of noise peaks

        # Find Hough space radii corresponding to each line
        pkctrs = self.find_peaks(hough_slice)
        radii = self.linhough_rad_from_bin(pkctrs)

        theta_max = self.linhough_theta_from_bin(thmax)

        labels, mindists = self.classify_points(xyz, arclens, theta_max, radii)

        return np.column_stack((xyz, labels, mindists, nncts)), (cu, cv)


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
