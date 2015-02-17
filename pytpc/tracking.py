"""Particle Tracking

This module contains code to track a particle in data.

"""

from __future__ import division, print_function
import numpy
from sklearn.cluster import DBSCAN

from pytpc.constants import *
import pytpc.simulation as sim

from pytpc.ukf import UnscentedKalmanFilter as UKF
#from filterpy.kalman import UnscentedKalmanFilter as UKF


def find_tracks(data, eps=20, min_samples=20):
    """ Applies the DBSCAN algorithm from scikit-learn to find tracks in the data.

    Arguments
    ---------

    data : An array of (x, y, z, hits) data points
    eps : The minimum distance between adjacent points in a cluster
    min_samples : The min number of points in a cluster

    Returns a list of numpy arrays containing the data points in each identified track.

    """
    xyz = data[:, 0:3]
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    dbs.fit(xyz)

    tracks = []
    for track in (numpy.where(dbs.labels_ == n)[0] for n in numpy.unique(dbs.labels_) if n != -1):
        tracks.append(data[track])

    return tracks


class Tracker:

    meas_dim = 3
    sv_dim = 6

    def __init__(self, particle, gas, efield, bfield, seed):
        self.particle = particle
        self.gas = gas
        self.efield = numpy.asarray(efield)
        self.bfield = numpy.asarray(bfield)
        self.kfilter = UKF(dim_x=self.sv_dim, dim_z=self.meas_dim, fx=self.update_state_vector,
                           hx=self.generate_measurement, dt=1.)

        self.kfilter.Q *= 1e-6
        self.kfilter.R *= 1e-2
        self.kfilter.x = seed
        self.kfilter.P *= 100

    def update_state_vector(self, state, dpos):
        """Find the next state vector.

        State vector is of the form (x, y, z, px, py, pz)

        """
        pos0 = state[0:3]
        mom0 = state[3:6]

        self.particle.position = pos0
        self.particle.momentum = mom0

        # tstep = pos_step / (self.particle.beta * c_lgt)

        new_state = sim.find_next_state(self.particle, self.gas, self.efield, self.bfield, dpos)
        self.particle.state_vector = new_state
        return numpy.hstack([self.particle.position, self.particle.momentum])

    def prev_state_vector(self, state):
        pos0 = state[0:3]
        mom0 = state[3:6]

        self.particle.position = pos0
        self.particle.momentum = mom0

        tstep = pos_step / (self.particle.beta * c_lgt)

        new_state = sim.find_prev_state(self.particle, self.gas, self.efield, -self.bfield, tstep)
        self.particle.state_vector = new_state
        return numpy.hstack([self.particle.position, self.particle.momentum])

    def generate_measurement(self, state):
        pos = state[0:3]
        # self.particle.position = pos
        return pos

    def track(self, meas):
        res, covar = self.kfilter.batch_filter(meas)
        return res, covar


if __name__ == '__main__':
    he_gas = sim.Gas(4, 2, 41.8, 150.)
    part = sim.Particle(mass_num=4, charge_num=2, energy_per_particle=2., azimuth=pi/5, polar=pi/4)
    e_field = [0., 0., 15e3]  # V/m
    b_field = [0., 0., 2.]  # T
    res_pos, res_azi, res_pol, res_time, res_en = sim.track(part, he_gas, e_field, b_field)
    tr = Tracker(part, he_gas, e_field, b_field)
    kf = tr.track(res_pos)
    print('Finished')