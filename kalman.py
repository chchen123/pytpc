import numpy
import scipy.constants
from numpy import sin, cos, tan, sqrt, log
from scipy.stats import threshold

# Constants

e_mc2 = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0]
p_mc2 = scipy.constants.physical_constants['proton mass energy equivalent in MeV'][0]
e_chg = scipy.constants.physical_constants['elementary charge'][0]
N_avo = scipy.constants.physical_constants['Avogadro constant'][0]
c_lgt = scipy.constants.physical_constants['speed of light in vacuum'][0]
pi = scipy.constants.pi
eps_0 = scipy.constants.physical_constants['electric constant'][0]

# Projectile Properties
charge_num = 2.  # unitless
mass_num = 4.  # unitless

charge = charge_num * e_chg  # C
mass = mass_num * p_mc2  # MeV
Bf = numpy.array([0., 0., 2.])  # Tesla
Ef = numpy.array([0., 0., 10000.])  # V/m

vel0 = mass*numpy.array([4e7, 2e7, 6e7])  # kg.m/s
pos0 = numpy.array([0., 0., 0.])  # m
tstep = 1e-9  # s
num_iters = 100

# Gas properties
gasZoA = 2./4.  # unitless
gasIonEn = 24.58   # in MeV
gasMass = 4.0026  # u
pressure = 300.  # torr
density = pressure / 760. * gasMass / 24040.  # in g/cm^3. 24040 is R*T at room temp.

MeVtokg = 1e-6 * e_chg
amuTokg = 1.66054e-27
amuToMeV = 931.494


def lorentz(vel, ef, bf, mass, charge):
    return charge*(ef + numpy.cross(vel, bf))


def bethe(projZ, projEn, gasZoA, gasIonEn):
    """ Find the stopping power of the gas.
    
    Arguments
    ---------
    projZ : The charge number of the projectile
    projEn : The energy of the projectile, in MeV/u
    gasZoA : The charge-to-mass-number ratio of the gas molecules
    gasIonEn : The ionization energy of the gas in MeV
    
    Returns: The stopping power in MeV / (g/cm^2) 
    
    """
    ne = N_avo * gasZoA * 1e6  # electrons /
    gamma = projEn/p_mc2 + 1  # this assumes MeV/u, which is why it uses just the proton mass
    beta_sq = 1 - 1/gamma**2
    vel_sq = beta_sq * c_lgt**2  # in (m/s)^2

    frnt = projZ**2 * e_chg**4 * ne / (4*pi * eps_0**2 * e_mc2 * MeVtokg * vel_sq)  # in J / (kg/m^2)
    lnt = log(2 * gamma**2 * e_mc2 * beta_sq / gasIonEn)
    dedx = frnt*(lnt - beta_sq)

    # Convert J -> eV -> MeV and kg/m^2 -> g/cm^2

    dedx = dedx/e_chg/1e6*10
    return dedx


def gamma_factor(v):
    """Returns the Lorentz gamma factor.
    
    The argument v may be a number or an array-like object.
    """
    vmag = numpy.linalg.norm(v)
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


def step_pos(state):
    """ Find the next step from the given state vector.
    
    The state vector should have the format:
        (x, y, z, vx, vy, vz)
        
    The units should be m and m/s.
    """
    pos = state[0:3]
    vel = state[3:6]
    en = (gamma_factor(vel) - 1)*mass / mass_num  # in MeV/u
    
    beta = beta_factor(en, mass)
    pos_step = tstep * beta * c_lgt / 100.  # in cm
    
    force = lorentz(vel, Ef, Bf, mass, charge)
    new_vel = vel + force/mass * tstep  # this is questionable w.r.t. relativity...
    stopping = bethe(charge_num, en, gasZoA, gasIonEn)  # in MeV / (g/cm^2)
    
    if stopping < 0:
        new_state = numpy.hstack((pos, numpy.zeros(3)))
        return new_state
    else:
        de = float(threshold(stopping*density*pos_step / mass, threshmin=1e-10))
        en = float(threshold(en - de, threshmin=0))
        
        new_beta = sqrt(1 - 1/(en/amuToMeV + 1)**2)
        new_vel *= new_beta / beta
        new_pos = pos + new_vel*tstep
        
        new_state = numpy.hstack((new_pos, new_vel))
        return new_state


# In[22]:

def track(pos0, en0, azi0, pol0):
    pos = []
    vel = []
    time = []
    en = []
    
    #beta = numpy.sqrt(1 - 1/(en0/amuToMeV + 1)**2)
    beta = beta_factor(en0, mass)
    v0 = numpy.array([beta*c_lgt*cos(azi0)*sin(pol0),
                      beta*c_lgt*sin(azi0)*sin(pol0),
                      beta*c_lgt*cos(pol0)])
    
    pos.append(pos0)
    vel.append(v0)
    en.append(en0)
    
    current_time = 0
    time.append(current_time)
    
    state = numpy.hstack((numpy.array(pos0), numpy.array(v0)))
    
    while True:
        state = step_pos(state)
        new_pos = state[0:3]
        new_vel = state[3:6]
        new_en = (gamma_factor(new_vel) - 1)*mass
        if new_en <= 0:
            break
        
        pos.append(new_pos)
        vel.append(new_vel)
        en.append(new_en)
        
        current_time += tstep
        time.append(current_time)
        
    return (pos, vel, en, time)

if __name__ == '__main__':
    track([0, 0, 0], 4, 1., 2.)