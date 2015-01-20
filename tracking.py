import numpy
from math import sin, cos, tan, log, sqrt, atan2, floor
from scipy.stats import threshold
from sklearn.cluster import DBSCAN
import copy

from constants import *
import kalman


class Gas:
    """ Describes a gas in the detector.
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

    @property
    def beta(self):
        en = self.energy
        m = self.mass
        return beta_factor(en, m)

    @property
    def gamma(self):
        return 1 / sqrt(1 - self.beta**2)

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

    meas_dim = 6
    sv_dim = 6

    def __init__(self, particle, gas, efield, bfield):
        self.particle = particle
        self.gas = gas
        self.efield = efield
        self.bfield = bfield
        self.kfilter = kalman.KalmanFilter(Tracker.sv_dim, Tracker.meas_dim,
                                          self.update_state_vector,
                                          self.jacobian)

    def update_state_vector(self, state):
        x0, y0, z0, k0, l0, phi0 = state
        r0 = 1/k0

        dphi = 0.005
        phi1 = phi0 + dphi

        dx = r0 * (cos(phi1) - cos(phi0))
        dy = r0 * (sin(phi1) - sin(phi0))
        dz = r0 * tan(l0) * dphi

        ds = sqrt(dx**2 + dy**2 + dz**2)

        en = recover_energy(k0, self.bfield, l0, self.particle.mass_num, self.particle.charge_num) / 4
        if en > 1e-4:
            self.particle.energy_per_particle = en
            beta = self.particle.beta
            dedx = bethe(self.particle, self.gas)  # * e_chg * 1e6  # transform from MeV/m to J/m
            try:
                gamma = self.particle.gamma
            except ZeroDivisionError:
                gamma = 1e10
            dk = k0 / (gamma * self.particle.mass * beta**2) * dedx * ds
    #         dk = (k0 * cos(l0)) / (beta * c_lgt * 2 * e_chg * numpy.linalg.norm(bfield)) * dedx * dphi
    #         dk = k0/(0.3 * numpy.linalg.norm(bfield) * beta) * dedx * dphi * 1e-3  # the 0.3 is in GeV/(m.T)
    #         print('en: {}, dedx: {}, dk: {}'.format(en, dedx, dk))
            k1 = k0 + dk
        else:
            k1 = k0

        r1 = 1 / k1

        x1 = x0 + dx
        y1 = y0 + dy
        z1 = z0 + dz
        l1 = l0

        return numpy.array([x1, y1, z1, 1/r1, l1, phi1])

    @staticmethod
    def jacobian(state):
        x, y, z, k, l, phi = state
        dphi = 0.005
        phip = phi+dphi
        res = numpy.array([[1, 0, 0, -dphi*tan(l)/k**2, dphi/(k*cos(l)**2), 0],
                           [0, 1, 0, -(-cos(phi)+cos(phip))/k**2, 0, (sin(phi)-sin(phip))/k],
                           [0, 0, 1, -(-sin(phi)+sin(phip))/k**2, 0, (-cos(phi)+cos(phip))/k],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        return res

    def track(self, meas):
        meas_sv, ctr = find_state_vector(meas)
        self.kfilter.apply(meas_sv)
        return self.kfilter


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
    ne = gas.electron_density_per_m3
    z = particle.charge_num
    I = gas.mean_exc_pot * 1e-6  # convert to MeV

    beta_sq = particle.beta**2

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
        
        new_beta = beta_factor(en, particle.mass)
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
    en.append(particle.energy_per_particle)
    
    while True:
        state = find_next_state(particle, gas, ef, bf)
        particle.state_vector = state
        if particle.energy == 0:
            print('Particle stopped')
            break
        
        pos.append(particle.position)
        azi.append(particle.azimuth)
        pol.append(particle.polar)
        en.append(particle.energy_per_particle)

        current_time += pos_step / (particle.beta * c_lgt)
        time.append(current_time)

        if particle.position[2] > 1 or sqrt(particle.position[0]**2 + particle.position[1]**2) > 0.275:
            print('Particle left chamber')
            break

    return pos, azi, pol, time, en


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
    he_gas = Gas(4, 2, 41.8, 150.)
    part = Particle(mass_num=4, charge_num=2, energy_per_particle=2., azimuth=pi/5, polar=pi/4)
    e_field = [0., 0., 15e3]  # V/m
    b_field = [0., 0., 2.]  # T
    res_pos, res_azi, res_pol, res_time, res_en = track(part, he_gas, e_field, b_field)
    tr = Tracker(part, he_gas, e_field, b_field)
    kf = tr.track(res_pos)
    print('Finished')