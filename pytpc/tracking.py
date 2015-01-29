"""Particle Tracking

This module contains code to track a particle in data.

"""

import numpy
from math import sqrt, floor
from sklearn.cluster import DBSCAN
import copy

from .constants import *
import pytpc.simulation as sim

from filterpy.kalman import UnscentedKalmanFilter as UKF


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
    dbs.fit(data)

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
        self.efield = efield
        self.bfield = bfield
        self.kfilter = UKF(dim_x=self.sv_dim, dim_z=self.meas_dim, dt=1e-8, fx=self.update_state_vector,
                           hx=self.generate_measurement)

        self.kfilter.Q *= 1e-6
        self.kfilter.R *= 1e-2
        self.kfilter.x = seed
        self.kfilter.P *= 100

    def update_state_vector(self, state, dt):
        """Find the next state vector.

        State vector is of the form (x, y, z, px, py, pz)

        """
        pos0 = state[0:3]
        mom0 = state[3:6]

        self.particle.position = pos0
        self.particle.momentum_mev = mom0

        new_state = sim.find_next_state(self.particle, self.gas, self.efield, self.bfield)
        self.particle.state_vector = new_state
        return numpy.hstack([self.particle.position, self.particle.momentum_mev])

    def jacobian(self, state):
        # pos = state[0:3]
        # mom = state[3:6]
        # self.particle.position = pos
        # self.particle.momentum = mom
        #
        # x, y, z, px, py, pz = state
        m = self.particle.mass_kg
        q = self.particle.charge
        bx, by, bz = self.bfield

        res = numpy.array([[1, 0, 0, 1/m, 0, 0],
                           [0, 1, 0, 0, 1/m, 0],
                           [0, 0, 1, 0, 0, 1/m],
                           [0, 0, 0, 1, q/m*bz, -q/m*by],
                           [0, 0, 0, -q/m*bz, 1, q/m*bx],
                           [0, 0, 0, q/m*by, -q/m*bx, 1]])
        return res


    def generate_measurement(self, state):
        pos = state[0:3]
        self.particle.position = pos
        return pos

    @staticmethod
    def measure_jacobian(state):
        x, y, z, px, py, pz = state

        print(state)

        return numpy.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, -((pos_step*px**2)/(px**2 + py**2 + pz**2)**(3/2))
                             + pos_step/sqrt(px**2 + py**2 + pz**2), -((pos_step*px*py)/(px**2 + py**2 + pz**2)**(3/2)),
                             -((pos_step*px*pz)/(px**2 + py**2 + pz**2)**(3/2))],
                            [0, 0, 0, -((pos_step*px*py)/(px**2 + py**2 + pz**2)**(3/2)),
                             -((pos_step*py**2)/(px**2 + py**2 + pz**2)**(3/2)) + pos_step/sqrt(px**2 + py**2 + pz**2),
                             -((pos_step*py*pz)/(px**2 + py**2 + pz**2)**(3/2))],
                            [0, 0, 0, -((pos_step*px*pz)/(px**2 + py**2 + pz**2)**(3/2)),
                             -((pos_step*py*pz)/(px**2 + py**2 + pz**2)**(3/2)), -((pos_step*pz**2)
                             /(px**2 + py**2 + pz**2)**(3/2)) + pos_step/sqrt(px**2 + py**2 + pz**2)]])

    def track(self, meas):
        res, covar = self.kfilter.batch_filter(meas)
        return res, covar

def find_circle_params(xs, ys):
    x1 = numpy.roll(xs, -1)
    x2 = copy.copy(xs)
    x3 = numpy.roll(xs, 1)
    y1 = numpy.roll(ys, -1)
    y2 = copy.copy(ys)
    y3 = numpy.roll(ys, 1)

    # Special cases: first and last. Use next two, or prev two
    x1[0], y1[0] = x2[0], y2[0]
    x2[0], y2[0] = x2[1], y2[1]
    x3[0], y3[0] = x2[2], y2[2]
    x1[-1], y1[-1] = x2[-1], y2[-1]
    x2[-1], y2[-1] = x2[-2], y2[-2]
    x3[-1], y3[-1] = x2[-3], y2[-3]

    curv = ((2 * numpy.abs(x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)) /
            numpy.sqrt(((x2-x1)**2+(y2-y1)**2) *
                       ((x2-x3)**2+(y2-y3)**2) *
                       ((x3-x1)**2+(y3-y1)**2)))

    # Find circle centers

    # first, mid, and last points
    mp = floor(x1.size / 2)
    mp2 = floor(mp/2)
    x1c = x2[0]
    y1c = y2[0]
    x2c = x2[mp2]
    y2c = y2[mp2]
    x3c = x2[mp]
    y3c = y2[mp]

    # midpoints of lines connecting the three points
    xm1 = (x1c + x2c) / 2
    ym1 = (y1c + y2c) / 2
    xm2 = (x2c + x3c) / 2
    ym2 = (y2c + y3c) / 2

    # slopes of the normal bisector lines, which intersect in the center
    sl1 = (x1c - x2c) / (y2c - y1c)
    sl2 = (x2c - x3c) / (y3c - y2c)

    # center point
    xc = -(-sl1*xm1 + sl2*xm2 + ym1 - ym2) / (sl1 - sl2)
    yc = -(-sl1*sl2*xm1 + sl1*sl2*xm2 + sl2*ym1 - sl1*ym2) / (sl1 - sl2)

    center = numpy.array((xc, yc))

    return curv, center


def find_phi(pts_in, center):
    pts = copy.copy(pts_in)
    pts -= center
    phi = numpy.arctan2(pts[:, 1], pts[:, 0])
    #phi -= phi[0]
    return numpy.unwrap(phi)


def find_pitch_angle(pos_in):
    pos = copy.copy(pos_in)
    pos_rot = numpy.roll(pos, 1, axis=0)
    pos_rot[0] = pos_rot[1]  # to fix the wrap-around
    diff = pos - pos_rot
    lambs = numpy.arctan2(diff[:, 2], numpy.sqrt(diff[:, 0]**2 + diff[:, 1]**2))
    return numpy.unwrap(lambs)


def find_state_vector(pts):
    """Returns the state vector from the position info of the track

    The state vector is (x, y, z, 1/R, lambda, phi)
    """

    curvs, ctr = find_circle_params(pts[:, 0], pts[:, 1])
    phis = find_phi(pts[:, 0:2], ctr)
    lmbs = find_pitch_angle(pts)
    found = numpy.vstack((curvs, lmbs, phis)).T
    return numpy.hstack((pts, found)), ctr


def recover_energy(curv, bfield, lambdas, mass_num, charge_num):
    """Recover the energies from the state vector parameters

    Arguments
    ---------
    curv : The curvatures (1/R) in inverse meters
    bfield : The magnetic field vector, in Tesla
    lambdas : The pitch angle of the helix at each point
    mass : The particle mass number, A
    charge : The particle's charge number, Z

    Returns the energies in MeV
    """
    mass = mass_num * p_mc2
    charge = charge_num * e_chg

    ps = 1/curv * numpy.linalg.norm(bfield) * numpy.sqrt(1+numpy.tan(lambdas)**2) * charge / e_chg / 1e6  # in MeV/c
    es = numpy.sqrt(ps**2 * c_lgt**2 + mass**2) - mass
    return es


if __name__ == '__main__':
    he_gas = sim.Gas(4, 2, 41.8, 150.)
    part = sim.Particle(mass_num=4, charge_num=2, energy_per_particle=2., azimuth=pi/5, polar=pi/4)
    e_field = [0., 0., 15e3]  # V/m
    b_field = [0., 0., 2.]  # T
    res_pos, res_azi, res_pol, res_time, res_en = sim.track(part, he_gas, e_field, b_field)
    tr = Tracker(part, he_gas, e_field, b_field)
    kf = tr.track(res_pos)
    print('Finished')