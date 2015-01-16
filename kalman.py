import numpy
import scipy.constants
from math import sin, cos, log, sqrt, atan2
from scipy.stats import threshold

# Constants

e_mc2 = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0]
p_mc2 = scipy.constants.physical_constants['proton mass energy equivalent in MeV'][0]
p_kg = scipy.constants.physical_constants['proton mass'][0]
e_chg = scipy.constants.physical_constants['elementary charge'][0]
N_avo = scipy.constants.physical_constants['Avogadro constant'][0]
c_lgt = scipy.constants.physical_constants['speed of light in vacuum'][0]
pi = scipy.constants.pi
eps_0 = scipy.constants.physical_constants['electric constant'][0]

pos_step = 1e-3  # in m
num_iters = 100

MeVtokg = 1e6 * e_chg / c_lgt**2
amuTokg = 1.66054e-27
amuToMeV = 931.494


class Gas:
    """ Describes a gas in the detector
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

    def get_density(self, unit='g/cm^3'):
        """ Return the density. Available units are g/cm^3, kg/m^3, and g/m^3
        """
        pressure = self.pressure / 760. * self.molar_mass / 24040.  # in g/cm^3. 24040 is R*T at room temp.
        if unit == 'g/cm^3':
            return pressure
        elif unit == 'kg/m^3':
            return pressure * 1000
        elif unit == 'g/m^3':
            return pressure * 1e6
        else:
            raise ValueError('Invalid units provided.')

    def get_electron_density(self, unit='cm^-3'):
        """ Get the electron density. Units are cm^-3 or m^-3
        """
        ne = N_avo * self.num_electrons * self.get_density(unit='g/cm^3') / self.molar_mass
        if unit == 'cm^-3':
            return ne
        elif unit == 'm^-3':
            return ne * 1e6
        else:
            raise ValueError('Invalid units provided.')


class Particle:
    """ Describes a beam particle for tracking.
    """

    def __init__(self, mass_num, charge_num, energy_per_particle, position=(0, 0, 0), azimuth=0, polar=0):
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

    def get_energy(self, unit='MeV', per_particle=False):
        """ Find the energy of the particle.

        The unit argument specifies the units to use. Options are:
            - MeV : The total energy in MeV
            - J : The total energy in joules

        The per_particle argument specifies if the total energy should be divided by the mass number.
        """

        if unit == 'MeV':
            res = self._energy
        elif unit == 'J':
            res = self._energy / 1e6 * e_chg
        else:
            raise ValueError('Invalid units provided.')

        if per_particle:
            res /= self.mass_num

        return res

    def get_mass(self, unit='MeV'):
        """ Find the mass of the particle.

        The unit argument specifies the units to use. Options are:
            - MeV : The total mass in MeV/c^2
            - kg : The total mass in kilograms
            - g : The total mass in grams
        """

        if unit == 'MeV':
            return self._mass
        elif unit == 'kg':
            return self._mass * MeVtokg
        elif unit == 'g':
            return self._mass * MeVtokg * 1000
        else:
            raise ValueError('Invalid units provided.')

    def get_velocity(self):
        """ Return the velocity of the particle, in m/s.
        """
        beta = self.beta()
        vel = numpy.array([beta*c_lgt*cos(self.azimuth)*sin(self.polar),
                           beta*c_lgt*sin(self.azimuth)*sin(self.polar),
                           beta*c_lgt*cos(self.polar)])
        return vel

    def beta(self):
        """ Returns beta, or v / c.
        """
        en = self.get_energy(per_particle=False)
        m = self.get_mass()
        return beta_factor(en, m)

    def gamma(self):
        """ Returns gamma for the particle.
        """
        return 1 / sqrt(1 - self.beta()**2)

    def get_state_vector(self):
        """ Generate the particle's state vector, as (x, y, z, en/u, azi, pol)
        """
        p = self.position  # for brevity
        res = numpy.array([p[0], p[1], p[2], self.get_energy(per_particle=True), self.azimuth, self.polar])
        return res

    def update_state(self, state_vector):
        """ Update the particle's parameters based on the provided state vector.

        The state vector should be (x, y, z, en/u, azi, pol)
        """
        new_position = state_vector[0:3]
        new_energy = state_vector[3]
        new_azimuth = state_vector[4]
        new_polar = state_vector[5]

        if new_energy < 0:
            # raise ValueError('Negative energy: {}'.format(new_energy))
            pass

        else:
            self.position = new_position
            self._energy = new_energy * self.mass_num
            self.azimuth = new_azimuth
            self.polar = new_polar


def lorentz(vel, ef, bf, charge):
    return charge*(ef + numpy.cross(vel, bf))


def bethe(particle, gas):
    """ Find the stopping power of the gas.
    
    Arguments
    ---------
    particle : A Particle object
    gas : A Gas object
    
    Returns: The stopping power in MeV/m
    
    """
    ne = gas.get_electron_density(unit='m^-3')
    z = particle.charge_num
    I = gas.mean_exc_pot * 1e-6  # convert to MeV

    beta_sq = particle.beta()**2

    try:
        frnt = ne * z**2 * e_chg**4 / (e_mc2 * MeVtokg * c_lgt**2 * beta_sq * 4 * pi * eps_0**2)
        lnt = log(2 * e_mc2 * beta_sq / (I * (1 - beta_sq)))
        dedx = frnt*(lnt - beta_sq)  # this should be in SI units, J/m

    except ZeroDivisionError:
        # This should only happen if beta == 0, so it should indicate that the particle has stopped.
        dedx = 0

    return dedx / e_chg * 1e-6  # converted to MeV/m


def gamma_factor(v):
    """Returns the Lorentz gamma factor.

    The argument v may be a number or an array-like object.
    """
    vmag = numpy.linalg.norm(v)
    if vmag > c_lgt:
        raise ValueError('Velocity was {}, which exceeds c.'.format(vmag))
    return 1/sqrt(1-vmag**2/c_lgt**2)


def beta_factor(en, mass):
    """ Returns beta, or v / c.

    The arguments should be in compatible units.

    Arguments
    ---------
    en : the relativistic kinetic energy
    mass : the rest mass
    """
    return (sqrt(en)*sqrt(en + 2*mass)) / (en + mass)


def find_next_state(particle, gas, ef, bf):
    """ Find the next step for the given particle and conditions.

    Returns the new state vector in the form (x, y, z, en/u, azi, pol)
    """

    en = particle.get_energy()
    vel = particle.get_velocity()
    charge = particle.charge
    pos = particle.position

    beta = particle.beta()
    if beta == 0:
        return particle.get_state_vector()
    tstep = pos_step / (beta * c_lgt)
    
    force = numpy.array(lorentz(vel, ef, bf, charge))
    new_vel = vel + force/particle.get_mass('kg') * tstep  # this is questionable w.r.t. relativity...
    stopping = bethe(particle, gas)  # in MeV/m
    de = float(threshold(stopping*pos_step, threshmin=1e-3))
    
    if stopping <= 0 or de == 0:
        new_state = particle.get_state_vector()
        new_state[3] = 0  # Set the energy to 0
        return new_state
    else:
        en = float(threshold(en - de, threshmin=0))
        
        new_beta = beta_factor(en, particle.get_mass())
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
    azi = []
    pol = []
    time = []
    en = []
    
    current_time = 0

    pos.append(particle.position)
    azi.append(particle.azimuth)
    pol.append(particle.polar)
    time.append(current_time)
    en.append(particle.get_energy(per_particle=True))
    
    while True:
        state = find_next_state(particle, gas, ef, bf)
        particle.update_state(state)
        if particle.get_energy() == 0:
            print('Particle stopped')
            break
        
        pos.append(particle.position)
        azi.append(particle.azimuth)
        pol.append(particle.polar)
        en.append(particle.get_energy(per_particle=True))

        current_time += pos_step / (particle.beta() * c_lgt)
        time.append(current_time)

        if particle.position[2] > 1 or sqrt(particle.position[0]**2 + particle.position[1]**2) > 0.275:
            print('Particle left chamber')
            break

    return pos, azi, pol, time, en

if __name__ == '__main__':
    he_gas = Gas(4, 2, 41.8, 150.)
    part = Particle(mass_num=4, charge_num=2, energy_per_particle=2., azimuth=pi/5, polar=pi/4)
    e_field = [0., 0., 15e3]  # V/m
    b_field = [0., 0., 2.]  # T
    res_pos, res_azi, res_pol, res_time, res_en = track(part, he_gas, e_field, b_field)
    print('Finished')