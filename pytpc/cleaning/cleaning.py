import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from .hough_wrapper import hough_line, hough_circle, nearest_neighbor_count
from ..fitting.mixins import PreprocessMixin
from ..utilities import Base, rot_matrix, tilt_matrix
from ..padplane import generate_pad_plane
from ..constants import pi, degrees


def linefunc(x, r, t):
    """Function of a line in the linear Hough space."""
    return (r - x * np.cos(t)) / np.sin(t)


class HoughCleaner(Base):
    """Provides functions to clean xyz data using the Hough transform and nearest-neighbor search.

    The different steps of the cleaning process are broken down into separate functions, so parts
    of the process can be applied independently (e.g. for testing purposes). The main function to use
    to clean a dataset is `clean`. That one will perform the full cleaning process.

    The class is initialized using a dictionary of configuration parameters. This might be loaded from
    a config file, for instance. The keys that must be in the dictionary are the same as the class's
    attributes given below. All keys for this class must be in a group called 'cleaning_config'.

    Parameters
    ----------
    config : dict-like
        A set of configuration parameters to initialize the object. See the attributes section below
        to find out what keys this dictionary must contain.

    Attributes
    ----------
    peak_width
        The number of bins to consider in each direction when finding the center of mass of
        peaks in the Hough space.
    linear_hough_max
        The largest radial distance to consider in the linear Hough transform. This sets the scale of the
        largest bin in the Hough space.
    linear_hough_nbins
        The number of bins to use in each dimension in the linear Hough transform.
    circle_hough_max
        The largest radial distance to consider in the circular Hough transform. This sets the scale of the
        largest bin in the Hough space.
    circle_hough_nbins
        The number of bins to use in each dimension in the circular Hough transform.
    min_pts_per_line
        If a line has fewer points than this near it, it will not be considered a true line.
    neighbor_radius
        The largest separation between two points that will still allow them to be considered neighbors.

    """
    def __init__(self, config):
        super().__init__(config)

        clean_conf = config['cleaning_config']

        self.peak_width = clean_conf['peak_width']

        self.linear_hough_max = clean_conf['linear_hough_max']
        self.linear_hough_nbins = clean_conf['linear_hough_nbins']
        self.circle_hough_max = clean_conf['circle_hough_max']
        self.circle_hough_nbins = clean_conf['circle_hough_nbins']

        self.min_pts_per_line = clean_conf['min_pts_per_line']
        self.neighbor_radius = clean_conf['neighbor_radius']

    def neighbor_count(self, xyz):
        """Count the number of neighbors each point has within radius `self.neighbor_radius`

        Parameters
        ----------
        xyz : array-like
            The 3D data to consider. The first three columns must correspond to the position coordinates.

        Returns
        -------
        nncts
            The number of neighbors that each point has.

        """
        xyz_cont = np.ascontiguousarray(xyz)
        nncts = nearest_neighbor_count(xyz_cont, self.neighbor_radius)
        return nncts

    def find_center(self, xyz):
        """Finds the center using the circular Hough transform.

        A nearest-neighbor cut will be applied to the data before finding the center.

        Parameters
        ----------
        xyz : array-like
            The 2D data to consider. The first two columns must be x and y positions.

        Returns
        -------
        cx, cy : float
            The x, y position of the found center.

        """
        xyz = np.ascontiguousarray(xyz[self.neighbor_count(xyz) > 1])
        return hough_circle(xyz, nbins=self.circle_hough_nbins, max_val=self.circle_hough_max)

    def find_arclen(self, xyz, cu, cv, extra_rotation=None):
        """Calculate the R-phi (arclength) position for each point with respect to the given center.

        Parameters
        ----------
        xyz : array-like
            The position data. Columns 0 and 1 are assumed to be x and y positions.
        cu, cv : float
            The x and y positions of the center, respectively.
        extra_rotation : float, optional
            If provided, the calculated theta values will be rotated by this amount (in radians) before returning.

        Returns
        -------
        ndarray
            The r-phi values.

        """
        rads = np.sqrt((xyz[:, 0] - cu)**2 + (xyz[:, 1] - cv)**2)
        thetas = np.arctan((xyz[:, 1] - cv) / (xyz[:, 0] - cu))

        if extra_rotation is not None:
            thetas = (thetas + (pi / 2) + extra_rotation) % pi - pi / 2

        return rads * thetas

    def find_linear_hough_space(self, zs, arclens):
        """Performs the linear Hough transform."""
        linear_hough_data = np.ascontiguousarray(np.column_stack((zs, arclens)))
        return hough_line(linear_hough_data, nbins=self.linear_hough_nbins, max_val=self.linear_hough_max)

    def find_hough_max_angle(self, lin_space):
        """Finds the maximum angle slice in the linear Hough space.

        The lines in the R-phi plot should all be parallel for a spiral-shaped track. Therefore we can find the
        angle corresponding to the highest max in the Hough space, and then look at a slice of the Hough space to
        find the radius corresponding to each line. This greatly reduces the number of lines to consider.

        Parameters
        ----------
        lin_space : ndarray
            The linear Hough space bins.

        Returns
        -------
        theta_max : float
            The angle of the found lines.
        hough_slice : ndarray
            A 1D slice of the Hough space at angle `theta_max`. The values are summed over a few angular bins to
            account for inaccuracies in `theta_max`.

        """
        # Find angle of max in Hough space
        maxidx = lin_space.ravel().argsort()  # List of peak indices in increasing order: look at last few in next step
        thmax = np.floor(np.unravel_index(maxidx[-5:], lin_space.shape)[0].mean()).astype('int')  # Index of average max
        theta_max = self.linhough_theta_from_bin(thmax)  # The value in angle units, rather than bins

        # Slice Hough space in angle dimension near max
        hough_slice = lin_space[thmax - 5:thmax + 5].sum(0)
        hough_slice[hough_slice < 60] = 0  # Use a threshold to get rid of noise peaks

        return theta_max, hough_slice

    def find_peaks(self, data):
        """Find the center of mass of the peaks in the 1D array `data` (e.g. the slice of the Hough space)."""
        def comparator(a, b):
            return np.greater_equal(a, b) & np.greater(a, 0)

        maxima = argrelextrema(data, comparator, order=2)[0]

        keepmask = data[maxima] != np.roll(data[maxima], -1)
        maxima = maxima[keepmask]

        pkctrs = np.empty_like(maxima, dtype='float64')

        for i, m in enumerate(maxima):
            first = max(m - self.peak_width, 0)
            last = min(m + self.peak_width, data.shape[0])
            pkpts = data[first:last]
            pklocs = np.arange(first, last)
            pkctrs[i] = (pkpts * pklocs).sum() / pkpts.sum()

        return pkctrs

    def linhough_theta_from_bin(self, t):
        return t * pi / self.linear_hough_nbins

    def linhough_rad_from_bin(self, r):
        return r * self.linear_hough_max * 2 / self.linear_hough_nbins - self.linear_hough_max

    def classify_points(self, xyz, arclens, theta, radii):
        """Identify which line each point belongs to, and find the distance from each point to the closest line.

        Parameters
        ----------
        xyz : array-like
            The data, with columns 0, 1, and 2 corresponding to x, y, and z.
        arclens : array-like
            The arclength (r-phi) values.
        theta : float
            The angle of the lines.
        radii : array-like
            The radius coordinate of each line.

        Returns
        -------
        labels : ndarray
            An integer label identifying which line each point is near. A value of -1 indicates that the point
            is not matched to any line.
        mindists : ndarray
            The distance from each point to the nearest line.

        """
        labels = np.full(xyz.shape[0], -1, dtype='int64')          # Group labels for each xyz point
        mindists = np.full(xyz.shape[0], np.inf, dtype='float64')  # Distance to closest line for each xyz point

        # Find min distances for each identified line
        for i, rad in enumerate(radii):
            dists = np.sqrt(np.square(linefunc(xyz[:, 2], rad, theta) - arclens))
            subset_mask = (dists < mindists)
            labels[subset_mask] = i
            mindists[subset_mask] = dists[subset_mask]

        for label_value in filter(lambda x: x != -1, np.unique(labels)):
            label_set = np.where(labels == label_value)[0]
            if len(label_set) < self.min_pts_per_line:
                labels[label_set] = -1

        return labels, mindists

    def clean(self, xyz):
        """Clean the data `xyz`.

        Parameters
        ----------
        xyz : array-like
            The data, with first three columns (x, y, z).

        Returns
        -------
        labels : ndarray
            An integer label identifying which line each point is near. A value of -1 indicates that the point
            is not matched to any line.
        mindists : ndarray
            The distance from each point to the nearest line.
        nn_counts : ndarray
            The number of neighbors each point has within distance `self.neighbor_radius`.
        (cu, cv) : tuple of floats
            The center of the event.

        """
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

        # Nearest neighbor cut
        nn_counts = self.neighbor_count(xyz)

        return labels, mindists, nn_counts, (cu, cv)


class EventCleaner(HoughCleaner, PreprocessMixin):
    def __init__(self, config):
        super().__init__(config)
        self.tilt = config['tilt'] * degrees
        self.vd = np.array(config['vd'])
        self.clock = config['clock']
        self.pad_rot_angle = config['pad_rot_angle'] * degrees
        self.padrotmat = rot_matrix(self.pad_rot_angle)
        self.beampads = []
        self.untilt_mat = tilt_matrix(self.tilt)
        self.pads = generate_pad_plane(self.pad_rot_angle)
        self.last_tb = config['cleaning_config']['last_tb']

    def process_event(self, evt):
        raw_xyz = evt.xyzs(pads=self.pads, peaks_only=True, return_pads=True, cg_times=True, baseline_correction=True)
        xyz = self.preprocess(raw_xyz, rotate_pads=False, last_tb=self.last_tb)

        cleaning_data = xyz[['u', 'v', 'w']].values

        labels, mindists, nn_counts, (cu, cv) = self.clean(cleaning_data)

        clean_xyz = np.column_stack((
            raw_xyz[np.where(raw_xyz[:, 2] < self.last_tb)],  # Remove dropped TBs to make dimensions match
            nn_counts,
            mindists,
        ))

        center = np.array([cu, cv, 0])
        center[1] -= np.tan(self.tilt) * 1000.
        cx, cy, _ = self.untilt_mat @ center

        return clean_xyz, (cx, cy)


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


def apply_clean_cut(raw_xyz_full, qthresh=40, nthresh=2, tbthresh=500):
    qcut = raw_xyz_full[:, -1] <= qthresh
    ncut = raw_xyz_full[:, -2] >= nthresh
    tbcut = raw_xyz_full[:, 2] <= tbthresh
    cut = qcut & ncut & tbcut
    return raw_xyz_full[cut, :5]
