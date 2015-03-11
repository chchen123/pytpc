"""Particle Tracking

This module contains code to track a particle in data.

"""

from __future__ import division, print_function
import numpy as np
from sklearn.cluster import DBSCAN
from functools import lru_cache

from pytpc.constants import *
import pytpc.simulation as sim
from pytpc.ukf import UnscentedKalmanFilter as UKF
from pytpc.padplane import generate_pad_plane


pads = generate_pad_plane()  # FIXME: Take rotation angle
pr = np.round(pads)


@lru_cache(maxsize=10240)
def find_adj(p, depth=0):
    """Recursively finds the neighboring pads to pad `p`.

    This function returns a set of neighboring pad numbers. The neighboring pads are determined by looking for
    all pads that (approximately) share a vertex with the given pad.

    .. note::
        Since only vertices are compared, this algorithm will not identify a neighboring pad that only shares a side,
        and not a vertex.

    The `depth` parameter allows recursion an arbitrary number of times. For instance, if `depth == 1`, the function
    will return the pads adjacent to pad `p` and those adjacent to its immediate neighbors.

    For efficiency, the results of this function are cached.

    Parameters
    ----------
    p : int
        The pad number whose neighbors you want to find
    depth : int
        The number of times to recurse

    Returns
    -------
    adj : set
        The adjacent pads

    """
    adj = set()
    for pt in pr[p]:
        i = np.argwhere(np.any(np.all(np.abs(pr[:, :] - pt) < np.array([2, 2]), axis=-1), axis=-1))
        adj |= set(i.ravel())

    if depth > 0:
        for a in adj.copy():
            adj |= find_adj(a, depth-1)

    return adj


def find_tracks(data, eps=20, min_samples=20):
    """Applies the DBSCAN algorithm from scikit-learn to find tracks in the data.

    Parameters
    ----------
    data : array-like
        An array of (x, y, z, hits) data points
    eps : number, optional
        The minimum distance between adjacent points in a cluster
    min_samples : number, optional
        The min number of points in a cluster

    Returns
    -------
    tracks : list
        A list of tracks. Each track is an ndarray of points.

    """
    xyz = data[:, 0:3]
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(xyz)

    tracks = []
    for track in (np.where(dbs.labels_ == n)[0] for n in np.unique(dbs.labels_) if n != -1):
        tracks.append(data[track])

    return tracks


class Tracker(object):
    """Class responsible for fitting particle tracks.

    This is basically a wrapper around the UnscentedKalmanFilter class that provides an interface between the physics
    simulation functions and the Kalman filter functions.

    Parameters
    ----------
    particle : Particle
        The particle to be tracked
    gas : Gas, or subclass
        The gas in the detector
    efield : array-like
        The electric field, in SI units
    bfield : array-like
        The magnetic field, in Tesla
    seed : array-like
        The initial seed for the Kalman filter. This is a guess for the state vector at time 0.

    Attributes
    ----------
    meas_dim : int
        The dimensionality of the measurements
    sv_dim : int
        The dimensionality of the state vectors
    """

    meas_dim = 3
    sv_dim = 6

    def __init__(self, particle, gas, efield, bfield, seed):
        self.particle = particle
        self.gas = gas
        self.efield = np.asarray(efield)
        self.bfield = np.asarray(bfield)
        self.kfilter = UKF(dim_x=self.sv_dim, dim_z=self.meas_dim, fx=self.update_state_vector,
                           hx=self.generate_measurement, dtx=self.find_timestep)

        self.kfilter.Q *= 1e-6
        self.kfilter.R *= 1e-2
        self.kfilter.x = seed
        self.kfilter.P *= 100

    def update_state_vector(self, state, dt):
        """Find the next state vector.

        This implements the f(x) prediction function for the Kalman filter.

        The state vector is of the form (x, y, z, px, py, pz).

        Parameters
        ----------
        state : array-like
            The current state
        dt : float
            The time step to the next state

        Returns
        -------
        ndarray
            The next state
        """
        pos0 = state[0:3]
        mom0 = state[3:6]

        self.particle.position = pos0
        self.particle.momentum = mom0

        # tstep = pos_step / (self.particle.beta * c_lgt)

        new_state = sim.find_next_state(self.particle, self.gas, self.efield, self.bfield, dt)
        self.particle.state_vector = new_state
        return np.hstack([self.particle.position, self.particle.momentum])

    def generate_measurement(self, state):
        """Convert a state vector to a measurement vector.

        This implements the h(x) measurement function for the Kalman filter.

        Parameters
        ----------
        state : array-like
            The state vector

        Returns
        -------
        ndarray
            The corresponding measurement vector
        """
        pos = state[0:3]
        # self.particle.position = pos
        return pos

    def find_timestep(self, sv, dpos):
        """Calculate a time step from a position step.

        This is done by dividing by the relativistic velocity.

        Parameters
        ----------
        sv : array-like
            The state vector
        dpos : float
            The position step

        Returns
        -------
        float
            The time step
        """
        self.particle.state_vector = sv
        return dpos / (self.particle.beta * c_lgt)

    def track(self, meas):
        """Run the tracker on the given data set.

        Parameters
        ----------
        meas : array-like
            The measured data points

        Returns
        -------
        res : ndarray
            The fitted state vectors
        covar : ndarray
            The covariance matrices of the state vectors
        times : ndarray
            The time at each step, for plotting on the x axis
        """
        res, covar, times = self.kfilter.batch_filter(meas)
        return res, covar, times