"""simulation.py

Contains code for simulating the tracking of particles in a TPC.

"""

import numpy
from math import atan2, log, sin, cos, sqrt
from scipy.stats import threshold

from constants import *
import relativity as rel


class Gas:
    """ Describes a gas in the detector.

    Attributes
    ----------
    molar_mass : The molar mass of the gas in g/mol
    num_electrons : The number of electrons per molecule, or the total Z
    mean_exc_pot : The mean excitation potential, as used in Bethe's formula, in eV
    pressure : The gas pressure in Torr
    density : The density in g/cm^3
    electron_density : The electron density per cm^3
    electron_density_per_m3 : The electron density per m^3

    """

    def __init__(self, molar_mass, num_electrons, mean_exc_pot, pressure):
        """ Initialize the instance.

        Arguments
        ---------
        molar_mass : Provided in g/mol
        num_electrons : Number of electrons per molecule, or the total Z
        mean_exc_pot : The mean excitation potential, as used in Bethe's formula, in eV
        pressure : The gas pressure in Torr
        """
        self.molar_mass = molar_mass
        self.num_electrons = num_electrons
        self.mean_exc_pot = mean_exc_pot
        self.pressure = pressure

    @property
    def density(self):
        """Density in g/cm^3"""
        return self.pressure / 760. * self.molar_mass / 24040.

    @property
    def electron_density(self):
        """Electron density per cm^3"""
        return N_avo * self.num_electrons * self.density / self.molar_mass

    @property
    def electron_density_per_m3(self):
        """Electron density per m^3"""
        return self.electron_density * 1e6


class Particle:
    """ Describes a beam particle for tracking.
    """

    def __init__(self, mass_num, charge_num, energy_per_particle=0, position=(0, 0, 0), azimuth=0, polar=0):
        """ Initialize the class.

        Arguments
        ---------
        mass_num : The A value of the particle (total number of nucleons)
        charge_num : The Z value of the particle
        energy_per_particle : Energy per nucleon, in MeV/u
        position : The initial position of the particle, as a list
        azimuth : The azimuthal angle of the particle's trajectory
        polar : The polar angle of the particle's trajectory

        """
        self.mass_num = mass_num
        self._mass = mass_num * p_mc2
        self.charge_num = charge_num
        self.charge = charge_num * e_chg
        self._energy = energy_per_particle * mass_num
        self.position = numpy.array(position)
        self.azimuth = azimuth
        self.polar = polar

    @property
    def energy(self):
        """The total energy in MeV"""
        return self._energy

    @energy.setter
    def energy(self, new):
        self._energy = new

    @property
    def energy_j(self):
        return self._energy * 1e6 * e_chg

    @energy_j.setter
    def energy_j(self, value):
        self._energy = value / 1e6 / e_chg

    @property
    def energy_per_particle(self):
        """The energy per particle in MeV/u"""
        return self._energy / self.mass_num

    @energy_per_particle.setter
    def energy_per_particle(self, new):
        self._energy = new * self.mass_num

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
        beta = self.beta
        vel = numpy.array([beta*c_lgt*cos(self.azimuth)*sin(self.polar),
                           beta*c_lgt*sin(self.azimuth)*sin(self.polar),
                           beta*c_lgt*cos(self.polar)])
        return vel

    @velocity.setter
    def velocity(self, new):
        vx, vy, vz = new
        self.energy = (rel.gamma(new) - 1) * self.mass
        self.azimuth = atan2(vy, vx)
        self.polar = atan2(sqrt(vx**2 + vy**2), vz)

    @property
    def momentum(self):
        """The momentum in kg.m/s"""
        return self.gamma * self.mass_kg * self.velocity

    @momentum.setter
    def momentum(self, new):
        px, py, pz = new
        self.energy_j = (numpy.sqrt(numpy.linalg.norm(new)**2 * c_lgt**2 + self.mass_kg**2 * c_lgt**4)
                         - self.mass_kg * c_lgt**2)
        self.azimuth = atan2(py, px)
        self.polar = atan2(sqrt(px**2 + py**2), pz)

    @property
    def beta(self):
        en = self.energy
        m = self.mass
        return rel.beta(en, m)

    @property
    def gamma(self):
        try:
            g = 1 / sqrt(1 - self.beta**2)
        except ZeroDivisionError:
            g = 1e100

        return g

    @property
    def state_vector(self):
        """The state vector of the particle, as (x, y, z, en/u, azi, pol).

        Setting to this will update every other property automatically.

        """
        p = self.position  # for brevity
        res = numpy.array([p[0], p[1], p[2], self.energy_per_particle, self.azimuth, self.polar])
        return res

    @state_vector.setter
    def state_vector(self, new):
        new_position = new[0:3]
        new_energy = new[3]
        new_azimuth = new[4]
        new_polar = new[5]

        if new_energy < 0:
            # raise ValueError('Negative energy: {}'.format(new_energy))
            pass

        else:
            self.position = new_position
            self._energy = new_energy * self.mass_num
            self.azimuth = new_azimuth
            self.polar = new_polar


def lorentz(vel, ef, bf, charge):
    """Calculate the Lorentz (electromagnetic) force.

    Arguments
    ---------
    vel : array-like
        The velocity
    ef : array-like
        The electric field
    bf : array-like
        The magnetic field
    charge : float or int
        The charge of the particle

    The units must be consistant to get a consistant result.

    Returns: The force, in whatever units the inputs generate.
    """

    try:
        return charge*(ef + numpy.cross(vel, bf))
    except ValueError:
        raise ValueError('vel, ef, bf must have length >= 2')


def bethe(particle, gas):
    """ Find the stopping power of the gas.

    Arguments
    ---------
    particle : A Particle object
    gas : A Gas object

    Returns: The stopping power in MeV/m

    """
    ne = gas.electron_density_per_m3
    z = particle.charge_num
    I = gas.mean_exc_pot * 1e-6  # convert to MeV

    beta_sq = particle.beta**2

    if beta_sq == 0.0:
        # The particle has stopped, so the dedx should be 0
        dedx = float('inf')
    elif beta_sq == 1.0:
        # This is odd, but then I guess dedx -> 0
        dedx = 0
    else:
        frnt = ne * z**2 * e_chg**4 / (e_mc2 * MeVtokg * c_lgt**2 * beta_sq * 4 * pi * eps_0**2)
        lnt = log(2 * e_mc2 * beta_sq / (I * (1 - beta_sq)))
        dedx = frnt*(lnt - beta_sq)  # this should be in SI units, J/m

    return dedx / e_chg * 1e-6  # converted to MeV/m


def find_next_state(particle, gas, ef, bf):
    """ Find the next step for the given particle and conditions.

    Returns the new state vector in the form (x, y, z, en/u, azi, pol)
    """

    en = particle.energy
    vel = particle.velocity
    charge = particle.charge
    pos = particle.position

    beta = particle.beta
    if beta == 0:
        return particle.state_vector
    tstep = pos_step / (beta * c_lgt)

    force = numpy.array(lorentz(vel, ef, bf, charge))
    new_vel = vel + force/particle.mass_kg * tstep  # this is questionable w.r.t. relativity...
    stopping = bethe(particle, gas)  # in MeV/m
    de = float(threshold(stopping*pos_step, threshmin=1e-3))

    if stopping <= 0 or de == 0:
        new_state = particle.state_vector
        new_state[3] = 0  # Set the energy to 0
        return new_state
    else:
        en = float(threshold(en - de, threshmin=0))

        new_beta = rel.beta(en, particle.mass)
        new_vel *= new_beta / beta
        new_pos = pos + new_vel*tstep

        new_azi = atan2(new_vel[1], new_vel[0])
        new_pol = atan2(sqrt(new_vel[0]**2 + new_vel[1]**2), new_vel[2])

        new_state = numpy.array([new_pos[0], new_pos[1], new_pos[2], en / particle.mass_num, new_azi, new_pol])
        return new_state


def track(particle, gas, ef, bf):
    """ Track the provided particle's trajectory.

    Arguments
    ---------
    particle : An instance of the Particle class

    Returns a tuple of (pos, vel, energy, time) lists

    """
    pos = []
    mom = []
    time = []
    en = []

    current_time = 0

    pos.append(particle.position)
    mom.append(particle.momentum)
    time.append(current_time)
    en.append(particle.energy_per_particle)

    while True:
        state = find_next_state(particle, gas, ef, bf)
        particle.state_vector = state
        if particle.energy == 0:
            print('Particle stopped')
            break

        pos.append(particle.position)
        mom.append(particle.momentum)
        en.append(particle.energy_per_particle)

        current_time += pos_step / (particle.beta * c_lgt)
        time.append(current_time)

        if particle.position[2] > 1 or sqrt(particle.position[0]**2 + particle.position[1]**2) > 0.275:
            print('Particle left chamber')
            break

    return pos, mom, time, en
