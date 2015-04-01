"""Special Relativity Functions

This module contains functions to calculate quantities of interest in special relativity.

"""

from __future__ import division, print_function
import numpy
from pytpc.constants import *
from math import sqrt, log, sinh, cosh, sin, cos, atan
import copy


def gamma(v):
    r"""Calculates the Lorentz gamma factor.

    Gamma is defined as 1 / sqrt(1 - beta**2) where beta = v / c.

    Parameters
    ----------
    v : array-like or number
        The velocity

    Raises
    ------
    ValueError
        If the magnitude of `v` is greater than the speed of light.
    """
    vmag = numpy.linalg.norm(v)
    if vmag >= c_lgt:
        raise ValueError('Velocity was {}, which exceeds c.'.format(vmag))
    return 1/sqrt(1-vmag**2/c_lgt**2)


def beta(en, mass):
    """Returns beta, or v / c.

    The arguments should be in compatible units.

    Parameters
    ----------
    en : int or float
        The relativistic kinetic energy
    mass : int or float
        The rest mass

    Raises
    ------
    ValueError
        If mass and energy are zero.
    """
    if en < 0.0 or mass < 0.0:
        raise ValueError('mass and energy must be positive')

    try:
        b = (sqrt(en)*sqrt(en + 2*mass)) / (en + mass)
    except ZeroDivisionError:
        raise ValueError('mass and energy cannot both be zero')

    if b > 1.0:
        b = 1.0
    return b


def elastic_scatter(proj, target, cm_angle, azi):
    """Perform the special relativity calculation needed for elastic scattering.

    This takes two particles and scatters them elastically at the given center-of-mass angle and azimuthal angle.
    These two angles are free parameters in the special relativity calculation, so they need to be given to the
    function.

    These equations are based on those at http://skisickness.com/2010/04/25/

    Parameters
    ----------
    proj : pytpc.simulation.Particle
        The projectile particle
    target : pytpc.simulation.Particle
        The target particle
    cm_angle : number
        The scattering angle in the center-of-momentum frame, in radians
    azi : number
        The final azimuthal angle for the scattered particles. The ejectile will be given this azimuthal angle, and
        the recoil particle will be given this angle + pi.

    Returns
    -------
    ejec : pytpc.simulation.Particle
        The ejectile particle
    recoil : pytpc.simulation.Particle
        The recoil particle
    """

    recoil = copy.copy(target)
    ejec = copy.copy(proj)
    m1 = target.mass
    m2 = proj.mass
    m3 = ejec.mass
    m4 = recoil.mass
    T1 = proj.energy  # Note: This is the kinetic energy only

    s = ((m1 + m2)**2 + 2 * m1 * T1)  # The Lorentz invariant. We don't need c's since the mass & mom are in MeV
    pcm  = sqrt(((s - m1**2 - m2**2)**2 - 4 * m1**2 * m2**2) / (4 * s))  # COM momentum of the reactants
    ppcm = sqrt(((s - m3**2 - m4**2)**2 - 4 * m3**2 * m4**2) / (4 * s))  # COM momentum of the products

    chi = log((sqrt(pcm**2 + m1**2) + pcm) / m1)  # rapidity of the COM frame

    T3 = sqrt(ppcm**2 + m3**2) * cosh(chi) + ppcm * cos(cm_angle) * sinh(chi) - m3  # ejectile KE
    T4 = sqrt(ppcm**2 + m4**2) * cosh(chi) - ppcm * cos(cm_angle) * sinh(chi) - m4  # recoil KE

    E3cm = sqrt(ppcm**2 + m3**2)  # ejectile COM total energy
    E4cm = sqrt(ppcm**2 + m4**2)  # recoil COM total energy

    pol3 = atan(ppcm * sin(cm_angle) / (sinh(chi) * E3cm + cosh(chi) * ppcm * cos(cm_angle)))  # ejectile lab polar ang
    pol4 = atan(ppcm * sin(cm_angle) / (sinh(chi) * E4cm + cosh(chi) * ppcm * cos(cm_angle)))  # recoil lab polar angle

    ejec.energy = T3
    ejec.polar = pol3
    ejec.azimuth = azi
    ejec.position = proj.position

    recoil.energy = T4
    recoil.polar = pol4
    recoil.azimuth = azi + pi
    recoil.position = proj.position

    return ejec, recoil


