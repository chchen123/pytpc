"""
simulation.py
=============

Contains code for simulating the tracking of particles in a TPC.

"""

from __future__ import division, print_function
cimport numpy as np
import numpy as np
import pandas as pd
from math import atan2, sin, cos, sqrt

import copy

from pytpc.constants import *
import pytpc.relativity as rel


cdef class Particle(object):
    """ Describes a beam particle for tracking.

    The state of the particle is stored internally as its energy and momentum. However, since most of the attributes
    are implemented as properties, setting any of them will update the internal state appropriately and make all other
    properties be up-to-date as well.

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

    cdef public int mass_num, charge_num
    cdef public double _mass, _energy, _mom_mag, _polar, _azimuth, charge
    cdef public np.ndarray position, _momentum

    def __init__(self, int mass_num, int charge_num, double energy_per_particle=0,
                 position=np.zeros(3, dtype=np.double), double azimuth=0., double polar=0.):
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
        self._polar = polar
        self._azimuth = azimuth

    property momentum:
        def __get__(self):
            """The particle's momentum in MeV/c"""
            return self._momentum

        def __set__(self, np.ndarray value):
            self._momentum = value
            px, py, pz = value
            self._mom_mag = sqrt(px**2 + py**2 + pz**2)
            self._energy = sqrt(self._mom_mag**2 + self.mass**2) - self.mass
            self._polar = atan2(sqrt(px**2 + py**2), pz)
            self._azimuth = atan2(py, px)

    property momentum_si:
        def __get__(self):
            """The particle's momentum in kg.m/s"""
            return self._momentum * 1e6 * e_chg / c_lgt

        def __set__(self, np.ndarray value):
            self.momentum = value * 1e-6 / e_chg * c_lgt

    property azimuth:
        def __get__(self):
            """The azimuthal angle of the trajectory in radians"""
            return self._azimuth

        def __set__(self, double value):
            self._azimuth = value
            azi = value
            pol = self._polar
            self._momentum = self._mom_mag * np.array([cos(azi) * sin(pol),
                                                       sin(azi) * sin(pol),
                                                       cos(pol)])

    property polar:
        def __get__(self):
            """The polar angle of the trajectory in radians"""
            return self._polar

        def __set__(self, double value):
            self._polar = value
            azi = self._azimuth
            pol = value
            self._momentum = self._mom_mag * np.array([cos(azi) * sin(pol),
                                                       sin(azi) * sin(pol),
                                                       cos(pol)])

    property energy:
        def __get__(self):
            """The total energy in MeV"""
            return self._energy

        def __set__(self, double value):
            self._mom_mag = sqrt((value + self.mass)**2 - self.mass**2)
            self._momentum = self._mom_mag * np.array([cos(self.azimuth) * sin(self.polar),
                                                          sin(self.azimuth) * sin(self.polar),
                                                          cos(self.polar)])
            self._energy = sqrt(self._mom_mag**2 + self.mass**2) - self.mass

    property energy_j:
        def __get__(self):
            """The total energy in J"""
            return self.energy * 1e6 * e_chg

        def __set__(self, double value):
            self.energy = value / 1e6 / e_chg

    property energy_per_particle:
        def __get__(self):
            """The energy per particle in MeV/u"""
            return self.energy / self.mass_num

        def __set__(self, double value):
            self.energy = value * self.mass_num

    property mass:
        def __get__(self):
            """The particle mass in MeV/c^2"""
            return self._mass

    property mass_kg:
        def __get__(self):
            """The particle mass in kg"""
            return self._mass * MeVtokg

    property velocity:
        def __get__(self):
            """The particle's velocity in m/s"""
            p_si = self.momentum * 1e6 / c_lgt * e_chg
            return p_si / (self.gamma * self.mass_kg)

        def __set__(self, np.ndarray value):
            gam = rel.gamma(value)
            p_si = gam * self.mass_kg * value
            self.momentum = p_si / e_chg * c_lgt * 1e-6

    property beta:
        def __get__(self):
            """The particle's beta, or v/c"""
            en = self.energy
            m = self.mass
            return rel.beta(en, m)

    property gamma:
        def __get__(self):
            """The particle's gamma, as defined in the Lorentz transform"""
            try:
                g = 1 / sqrt(1 - self.beta**2)
            except ZeroDivisionError:
                g = 1e100

            return g

    property state_vector:
        def __get__(self):
            """The state vector of the particle, as (x, y, z, px, py, pz).

            Setting to this will update every other property automatically.
            """
            return np.hstack([self.position, self.momentum])

        def __set__(self, np.ndarray value):
            self.position = value[0:3]
            self.momentum = value[3:6]


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
    Bx, By, Bz = bf

    fx = charge * (ex + vy*Bz - vz*By)
    fy = charge * (ey + vz*Bx - vx*Bz)
    fz = charge * (ez + vx*By - vy*Bx)

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
    dpos = tstep * particle.beta * c_lgt
    stopping = gas.energy_loss(particle.energy, particle.mass_num, particle.charge_num)  # in MeV/m
    de = float(threshold(stopping*dpos, threshmin=0))

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


def track(particle, gas, ef, bf, final_energy=0.0):
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
        if particle.energy <= final_energy:
            # print('Reached final energy')
            done = True

        pos.append(particle.position)
        mom.append(particle.momentum)
        en.append(particle.energy_per_particle)
        de.append(en[-2] - en[-1])
        azi.append(particle.azimuth)
        pol.append(particle.polar)

        current_time += tstep
        time.append(current_time)

        if not(0 <= particle.position[2] <= 1) or sqrt(particle.position[0]**2 + particle.position[1]**2) > 0.275:
            # print('Particle left chamber')
            done = True

    res_keys = ['x', 'y', 'z', 'px', 'py', 'pz', 'time', 'en', 'de', 'azi', 'pol']

    pos = np.array(pos)
    mom = np.array(mom)
    res = np.hstack((pos, mom, np.array([time, en, de, azi, pol]).T))

    tracks = pd.DataFrame(res, columns=res_keys)
    return tracks


def simulate_elastic_scattering_track(proj, target, gas, ef, bf, interact_energy, cm_angle, azimuth):
    """Simulate an elastic scattering event with the given particles and parameters.

    The projectile will be tracked in the detector until it has total energy equal to `interact_energy`. Then, an
    elastic scattering interaction will happen with the provided COM angle and final azimuthal angle.

    Parameters
    ----------
    proj : Particle
        The projectile. It must have non-zero energy.
    target : Particle
        The target. It must have zero energy (at least for now)
    gas : pytpc.gases.Gas or subclass
        The gas in the detector. If it's a subclass, make sure it is compatible with both the target and the
        projectile.
    ef : array-like
        The electric field, in SI units
    bf : array-like
        The magnetic field, in Tesla
    interact_energy : number
        The energy at which to stop tracking the projectile and perform the interaction
    cm_angle : number
        The center-of-momentum angle for the relativistic scattering calculation
    azimuth : number
        The final azimuthal angle of the ejectile particle

    Returns
    -------
    pandas.DataFrame
        The concatenated outputs of the `track` function for each particle

    Raises
    ------
    ValueError
        If the projectile has zero energy
    NotImplementedError
        If the target has nonzero energy. In the future, the code could be changed to allow a moving target particle,
        but it seems unnecessary for now.

    See Also
    --------
    pytpc.relativity.elastic_scatter

    """

    if proj.energy == 0:
        raise ValueError('Projectile must have energy > 0')
    if target.energy > 0.0:
        raise NotImplementedError('Target energy must be zero (for now at least)')

    # Track projectile until the collision
    proj_track = track(proj, gas, ef, bf, interact_energy)
    if not(0 <= proj.position[2] <= 1) or sqrt(proj.position[0]**2 + proj.position[1]**2) > 0.275:
        # The particle left the chamber before interacting
        return proj_track

    reaction_time = proj_track.time.max()

    ejec, recoil = rel.elastic_scatter(proj, target, cm_angle, azimuth)

    ejec_track = track(ejec, gas, ef, bf)
    ejec_track.time += reaction_time
    recoil_track = track(recoil, gas, ef, bf)
    recoil_track.time += reaction_time

    return pd.concat((proj_track, ejec_track, recoil_track), ignore_index=True)


def drift_velocity_vector(vd, efield, bfield, tilt):
    r"""Returns the drift velocity vector.

    This includes the effects due to the Lorentz angle from the non-parallel electric and magnetic fields.

    Parameters
    ----------
    vd : number
        The drift velocity, in cm/us.
    efield : number
        The electric field magnitude, in SI units.
    bfield : number
        The magnetic field magnitude, in Tesla.
    tilt : number
        The angle between the electric and magnetic fields, in radians.

    Returns
    -------
    ndarray
        The drift velocity vector

    Notes
    -----
    This formula is a steady-state solution to the Lorentz force equation with a damping term:

    ..  math:: m \dot{\mathbf{v}} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B}) - \beta \mathbf{v}

    The form of the damping term is not important since I'm not actually deriving the solution here. References [1]_
    state that the solution is

    .. math::
        \mathbf{v}_D = \frac{\mu E}{1 + \omega^2\tau^2} [ \hat E + \omega\tau \hat{E}\times\hat{B}
            + \omega^2\tau^2(\hat{E}\cdot\hat{B})\hat{B} ]

    If we let

    .. math::
        \omega = \frac{eB}{m}; \quad \tau = \frac{m v_D}{eE}; \quad \mu = \frac{e\tau}{m}

    and assume that the electric field is along z and the magnetic field is tilted away from z at an angle `tilt` along
    the y direction, then this reduces to the form implemented here.

    References
    ----------
    ..  [1] T. Lohse and W. Witzeling, in *Instrumentation in High Energy Physics*, edited by F. Sauli
        (World Scientific, Singapore, 1992), p. 106.

    """

    ot = bfield / efield * vd * 1e4  # omega * tau in formula. The numerical factor converts cm/us -> m/s

    front = vd / (1 + ot**2)

    xcomp = -front * ot * sin(tilt)
    ycomp = front * ot**2 * cos(tilt) * sin(tilt)
    zcomp = front * (1 + ot**2 * cos(tilt)**2)

    return np.array([xcomp, ycomp, zcomp])  # in cm/us, as long as vd had the right units