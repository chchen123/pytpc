import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from .hough_wrapper import hough_line, hough_circle, nearest_neighbor_count
from ..constants import pi


def linefunc(x, r, t):
    """Function of a line in the linear Hough space."""
    return (r - x * np.cos(t)) / np.sin(t)


class HoughCleaner(object):
    def __init__(self, config):
        self.peak_width = config['peak_width']

        self.linear_hough_max = config['linear_hough_max']
        self.linear_hough_nbins = config['linear_hough_nbins']
        self.circle_hough_max = config['circle_hough_max']
        self.circle_hough_nbins = config['circle_hough_nbins']

        self.max_distance_from_line = config['max_distance_from_line']
        self.min_pts_per_line = config['min_pts_per_line']
        self.min_num_neighbors = config['min_num_neighbors']
        self.neighbor_radius = config['neighbor_radius']

    def neighbor_cut(self, xyz):
        xyz_cont = np.ascontiguousarray(xyz)
        nncts = nearest_neighbor_count(xyz_cont, self.neighbor_radius)
        return xyz[nncts > self.min_num_neighbors]

    def find_center(self, xyz):
        return hough_circle(xyz, nbins=self.circle_hough_nbins, max_val=self.circle_hough_max)

    def find_arclen(self, xyz, cu, cv, extra_rotation=None):
        rads = np.sqrt((xyz[:, 0] - cu)**2 + (xyz[:, 1] - cv)**2)
        thetas = np.arctan((xyz[:, 1] - cv) / (xyz[:, 0] - cu))

        if extra_rotation is not None:
            thetas = (thetas + (pi / 2) + extra_rotation) % pi - pi / 2

        return rads * thetas

    def find_linear_hough_space(self, zs, arclens):
        linear_hough_data = np.ascontiguousarray(np.column_stack((zs, arclens)))
        return hough_line(linear_hough_data, nbins=self.linear_hough_nbins, max_val=self.linear_hough_max)

    def find_hough_max_angle(self, lin_space):
        # Find angle of max in Hough space
        maxidx = lin_space.ravel().argsort()  # List of peak indices in increasing order: look at last few in next step
        thmax = np.floor(np.unravel_index(maxidx[-5:], lin_space.shape)[0].mean()).astype('int')  # Index of average max
        theta_max = self.linhough_theta_from_bin(thmax)  # The value in angle units, rather than bins

        # Slice Hough space in angle dimension near max
        hough_slice = lin_space[thmax - 5:thmax + 5].sum(0)
        hough_slice[hough_slice < 60] = 0  # Use a threshold to get rid of noise peaks

        return theta_max, hough_slice

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
        # Nearest neighbor cut
        xyz = self.neighbor_cut(xyz)

        # Find center
        cu, cv = self.find_center(xyz)

        # Find r-theta (arclength)
        arclens = self.find_arclen(xyz, cu, cv)

        # Perform linear Hough transform
        lin_space = self.find_linear_hough_space(xyz[:, 2], arclens)

        # Find angle of max in Hough space and make a slice around that angle
        theta_max, hough_slice = self.find_hough_max_angle(lin_space)

        # Find Hough space radii corresponding to each line
        pkctrs = self.find_peaks(hough_slice)
        radii = self.linhough_rad_from_bin(pkctrs)

        # Identify which line (if any) each point belongs to
        labels, mindists = self.classify_points(xyz, arclens, theta_max, radii)

        return np.column_stack((xyz, labels, mindists)), (cu, cv)


def nn_remove_noise(data, radius=40, num_neighbors=10):
    if isinstance(data, pd.DataFrame):
        xyz = np.ascontiguousarray(data.values)
    else:
        xyz = np.ascontiguousarray(data)

    counts = nearest_neighbor_count(xyz, radius)

    noise = counts < num_neighbors

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
