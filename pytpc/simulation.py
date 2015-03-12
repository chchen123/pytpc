"""
simulation.py
=============

Contains code for simulating the tracking of particles in a TPC.

"""

from __future__ import division, print_function
import numpy as np
from math import atan2, sin, cos, sqrt

import copy

from pytpc.constants import *
import pytpc.relativity as rel
from pytpc.padplane import generate_pad_plane, inner_rad, inner_tri_base, inner_tri_height, tri_base, tri_height
from pytpc.utilities import skew_matrix, rot_matrix


class Particle(object):
    """ Describes a beam particle for tracking.

    Parameters
    ----------
    mass_num : int
        The A value of the particle (total number of nucleons)
    charge_num : int
        The Z value of the particle
    energy_per_particle : float
        Energy per nucleon, in MeV/u
    position : array-like
        The initial position of the particle
    azimuth : float
        The azimuthal angle of the particle's trajectory, in radians
    polar : float
        The polar angle of the particle's trajectory, in radians
    """

    def __init__(self, mass_num, charge_num, energy_per_particle=0, position=(0, 0, 0), azimuth=0, polar=0):
        self.mass_num = mass_num
        self._mass = mass_num * p_mc2
        self.charge_num = charge_num
        self.charge = charge_num * e_chg
        self.position = np.array(position)
        self._energy = energy_per_particle * self.mass_num
        self._mom_mag = sqrt((self._energy + self.mass)**2 - self.mass**2)
        self._momentum = self._mom_mag * np.array([cos(azimuth) * sin(polar),
                                                      sin(azimuth) * sin(polar),
                                                      cos(polar)])

    @property
    def momentum(self):
        """The particle's momentum in MeV/c"""
        return self._momentum

    @momentum.setter
    def momentum(self, new):
        self._momentum = new
        self._mom_mag = np.linalg.norm(new)
        self._energy = sqrt(self._mom_mag**2 + self.mass**2) - self.mass

    @property
    def momentum_si(self):
        """The particle's momentum in kg.m/s"""
        return self._momentum * 1e6 * e_chg / c_lgt

    @momentum_si.setter
    def momentum_si(self, value):
        self.momentum = value * 1e-6 / e_chg * c_lgt

    @property
    def azimuth(self):
        """The azimuthal angle of the trajectory in radians"""
        px, py, pz = self.momentum
        return atan2(py, px)

    @property
    def polar(self):
        """The polar angle of the trajectory in radians"""
        px, py, pz = self.momentum
        return atan2(sqrt(px**2 + py**2), pz)

    @property
    def energy(self):
        """The total energy in MeV"""
        return self._energy

    @energy.setter
    def energy(self, new):
        self._mom_mag = sqrt((new + self.mass)**2 - self.mass**2)
        self._momentum = self._mom_mag * np.array([cos(self.azimuth) * sin(self.polar),
                                                      sin(self.azimuth) * sin(self.polar),
                                                      cos(self.polar)])
        self._energy = sqrt(self._mom_mag**2 + self.mass**2) - self.mass

    @property
    def energy_j(self):
        """The total energy in J"""
        return self.energy * 1e6 * e_chg

    @energy_j.setter
    def energy_j(self, value):
        self.energy = value / 1e6 / e_chg

    @property
    def energy_per_particle(self):
        """The energy per particle in MeV/u"""
        return self.energy / self.mass_num

    @energy_per_particle.setter
    def energy_per_particle(self, new):
        self.energy = new * self.mass_num

    @property
    def mass(self):
        """The particle mass in MeV/c^2"""
        return self._mass

    @property
    def mass_kg(self):
        """The particle mass in kg"""
        return self._mass * MeVtokg

    @property
    def velocity(self):
        """The particle's velocity in m/s"""
        p_si = self.momentum * 1e6 / c_lgt * e_chg
        return p_si / (self.gamma * self.mass_kg)

    @velocity.setter
    def velocity(self, new):
        gam = rel.gamma(new)
        p_si = gam * self.mass_kg * new
        self.momentum = p_si / e_chg * c_lgt * 1e-6

    @property
    def beta(self):
        """The particle's beta, or v/c"""
        en = self.energy
        m = self.mass
        return rel.beta(en, m)

    @property
    def gamma(self):
        """The particle's gamma, as defined in the Lorentz transform"""
        try:
            g = 1 / sqrt(1 - self.beta**2)
        except ZeroDivisionError:
            g = 1e100

        return g

    @property
    def state_vector(self):
        """The state vector of the particle, as (x, y, z, px, py, pz).

        Setting to this will update every other property automatically.
        """
        return np.hstack([self.position, self.momentum])

    @state_vector.setter
    def state_vector(self, new):
        self.position = new[0:3]
        self.momentum = new[3:6]


def lorentz(vel, ef, bf, charge):
    """Calculate the Lorentz (electromagnetic) force.

    This is defined as

    ..  math::
        \\mathbf{F} = q(\\mathbf{E} + \\mathbf{v} \\times \\mathbf{B})

    ..  note::
        The units of the inputs must be consistant to get a consistant result.

    Parameters
    ----------
    vel : array-like
        The velocity
    ef : array-like
        The electric field
    bf : array-like
        The magnetic field
    charge : float or int
        The charge of the particle

    Returns
    -------
    force : array-like
        The electromagnetic force, in whatever units the inputs generate
    """

    vx, vy, vz = vel
    ex, ey, ez = ef
    bx, by, bz = bf

    fx = charge * (ex + vy*bz - vz*by)
    fy = charge * (ey + vz*bx - vx*bz)
    fz = charge * (ez + vx*by - vy*bx)

    return np.array([fx, fy, fz])


def threshold(value, threshmin=0.):
    """Applies a threshold to the given value.

    Any value less than threshmin will be replaced by threshmin.

    Parameters
    ----------
    value : number
        The value to be thresholded
    threshmin : number
        The minimum threshold value

    """
    if value < threshmin:
        return threshmin
    else:
        return value


def find_next_state(particle, gas, ef, bf, tstep):
    """ Find the next step for the given particle and conditions.

    Parameters
    ----------
    particle : Particle
        The particle to be tracked. It will **not** be changed by this function.
    gas : Gas
        The gas in the detector
    ef : array-like
        The electric field
    bf : array-like
        The magnetic field
    tstep : float
        The time step to the next state

    Returns
    -------
    state_vector : ndarray
        The new state vector. The format of this is (x, y, z, px, py, pz). This
        can be provided to an instance of the Particle class to update its state
        in a tracking loop.
    """

    en = particle.energy
    mom = particle.momentum_si
    vel = particle.velocity
    charge = particle.charge

    beta = particle.beta
    if beta == 0:
        return particle.state_vector

    force = lorentz(vel, ef, bf, charge)
    new_mom = mom + force * tstep
    stopping = gas.energy_loss(particle.energy, particle.mass_num, particle.charge_num)  # in MeV/m
    de = float(threshold(stopping*pos_step, threshmin=0))

    if stopping <= 0 or de == 0:
        new_state = particle.state_vector
        new_state[3:6] = [0, 0, 0]  # Set the momentum to 0
        return new_state
    else:
        en = float(threshold(en - de, threshmin=0.))

        new_particle = copy.copy(particle)
        new_particle.momentum_si = new_mom
        new_particle.energy = en

        new_vel = new_particle.velocity
        new_particle.position += new_vel * tstep

        return new_particle.state_vector


def track(particle, gas, ef, bf):
    """ Track the provided particle's trajectory until it stops.

    This can be used to simulate a particle's motion through the detector.

    Parameters
    ----------
    particle : Particle
        The particle to be tracked.
    gas : Gas
        The gas in the detector
    ef : array-like
        The electric field
    bf : array-like
        The magnetic field

    Returns
    -------
    dict
        The results, as a dictionary. The keys in this dictionary are shown below.
            - pos: The position of the particle at each step
            - mom: The momentum at each step
            - time: The time at each step
            - en: The energy per particle at each step
            - de: The change in energy at each step
            - azi: The azimuthal angle of the trajectory at each step
            - pol: The polar angle of the trajectory at each step

    """
    pos = []
    mom = []
    time = []
    en = []
    de = []
    azi = []
    pol = []

    current_time = 0

    pos.append(particle.position)
    mom.append(particle.momentum)
    time.append(current_time)
    en.append(particle.energy_per_particle)
    de.append(0.0)
    azi.append(particle.azimuth)
    pol.append(particle.polar)

    done = False
    while not done:
        tstep = pos_step / (particle.beta * c_lgt)
        state = find_next_state(particle, gas, ef, bf, tstep)
        particle.state_vector = state
        if particle.energy == 0:
            print('Particle stopped')
            done = True

        pos.append(particle.position)
        mom.append(particle.momentum)
        en.append(particle.energy_per_particle)
        de.append(en[-2] - en[-1])
        azi.append(particle.azimuth)
        pol.append(particle.polar)

        current_time += tstep
        time.append(current_time)

        if particle.position[2] > 1 or sqrt(particle.position[0]**2 + particle.position[1]**2) > 0.275:
            print('Particle left chamber')
            done = True

    res_keys = ['pos', 'mom', 'time', 'en', 'de', 'azi', 'pol']
    res_vals = map(np.array, (pos, mom, time, en, de, azi, pol))

    return dict(zip(res_keys, res_vals))


_leftskewmat = skew_matrix(-60.*degrees)
_rightskewmat = skew_matrix(60.*degrees)
_rotneg120mat = rot_matrix(-120*degrees)
_rotpos120mat = rot_matrix(120*degrees)
_rotneg240mat = rot_matrix(-240*degrees)
_rotpos240mat = rot_matrix(240*degrees)
_idmat = np.eye(2)


def _two_pt_line(x, x1, y1, x2, y2):

    return (y2-y1)/(x2-x1) * (x-x1) + y1

# NOTE: The function below does not work, but I'm keeping it here as a starting point
#
# def find_pad_coords_vec(x, y):
#
#     coords = np.vstack((x, y)).T  # Make x-y pairs
#
#     # Which third? Rotate if necessary
#     angle = np.arctan2(y, x) + np.where(y < 0, 2*pi, 0)
#
#     trans = np.zeros((coords.shape[0], 2, 2))
#     trans_temp = np.where(np.logical_and(120*degrees <= angle, angle < 240*degrees), _rotneg120mat, 0)
#     trans += trans_temp
#     trans_temp = np.where(angle >= 240*degrees, _rotneg240mat, 0)
#     trans += trans_temp
#     trans_temp = np.where(angle < 120*degrees, _idmat, 0)
#     trans += trans_temp
#
#     trans_inv = np.zeros((coords.shape[0], 2, 2))
#     trans_temp = np.where(np.logical_and(120*degrees <= angle, angle < 240*degrees), _rotpos120mat, 0)
#     trans_inv += trans_temp
#     trans_temp = np.where(angle >= 240*degrees, _rotpos240mat, 0)
#     trans_inv += trans_temp
#     trans_temp = np.where(angle < 120*degrees, _idmat, 0)
#     trans_inv += trans_temp
#     del trans_temp
#
#     rot = np.tensordot(trans, coords, axes=(2, 1))
#
#     rect = np.tensordot(_rightskewmat, rot, axes=(2, 1))
#
#     b = np.where(np.logical_and(np.abs(rect[:, 0]) < inner_rad * tri_base, np.abs(rect[:, 1]) < inner_rad * tri_height),
#                  inner_tri_base, tri_base)
#     h = np.where(b == inner_tri_base, inner_tri_height, tri_height)
#     t = np.vstack((b, h)).T
#
#     v1 = np.floor(rect / t) * t
#     v2 = v1 + t
#
#     tpl = _two_pt_line(rect[:, 0], v1[:, 0], v1[:, 1], v2[:, 0], v2[:, 1])
#     harr = np.zeros_like(t)
#     harr[:, 1] = h
#     barr = np.zeros_like(t)
#     barr[:, 0] = b
#
#     v3 = np.where(rect[1] > tpl, v1 + harr, v1 + barr)
#
#     res = np.tensordot(_leftskewmat, (v1+v2+v3)/3, axes=(2, 1))
#     res = np.tensordot(trans_inv, res)
#     return res


def find_pad_coords(x, y):

    coords = np.array([x, y])

    # Which third? Rotate if necessary
    angle = atan2(y, x)
    if y < 0:
        angle += 2*pi

    if 120*degrees <= angle < 240*degrees:
        rot = np.dot(_rotneg120mat, coords)
        finalrot = _rotpos120mat
    elif angle >= 240*degrees:
        rot = np.dot(_rotneg240mat, coords)
        finalrot = _rotpos240mat
    else:
        rot = coords
        finalrot = _idmat

    rect = np.dot(_rightskewmat, rot)

    # Inner or outer?
    if abs(rect[0]) < inner_rad * tri_base and abs(rect[1]) < inner_rad * tri_height:
        b = inner_tri_base
        h = inner_tri_height
        t = np.array([b, h])
    else:
        b = tri_base
        h = tri_height
        t = np.array([b, h])

    v1 = np.floor(rect / t) * t
    v2 = v1 + t

    if rect[1] > _two_pt_line(rect[0], v1[0], v1[1], v2[0], v2[1]):
        v3 = v1 + np.array([0, h])
    else:
        v3 = v1 + np.array([b, 0])

    res = np.dot(_leftskewmat, (v1+v2+v3)/3)
    res = np.dot(finalrot, res)
    return res
