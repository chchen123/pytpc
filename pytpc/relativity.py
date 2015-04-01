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

    recoil = copy.copy(target)
    ejec = copy.copy(proj)
    m1 = target.mass
    m2 = proj.mass
    m3 = ejec.mass
    m4 = recoil.mass
    T1 = proj.energy

    s = (m1 + m2)**2 * c_lgt**2 + 2 * m1 * T1
    pcm  = sqrt(((s - m1**2 * c_lgt**2 - m2**2 * c_lgt**2)**2 - 4 * m1**2 * m2**2 * c_lgt**4) / (4 * s))
    ppcm = sqrt(((s - m3**2 * c_lgt**2 - m4**2 * c_lgt**2)**2 - 4 * m3**2 * m4**2 * c_lgt**4) / (4 * s))

    chi = log((sqrt(pcm**2 * c_lgt**2 + m1**2 * c_lgt**4) + pcm * c_lgt) / (m1 * c_lgt**2))

    T3 = (sqrt(ppcm**2 * c_lgt**2 + m3**2 * c_lgt**4) * cosh(chi) + ppcm * c_lgt * cos(cm_angle) * sinh(chi)
          - m3 * c_lgt**2)
    T4 = (sqrt(ppcm**2 * c_lgt**2 + m4**2 * c_lgt**4) * cosh(chi) - ppcm * c_lgt * cos(cm_angle) * sinh(chi)
          - m4 * c_lgt**2)

    E3cm = sqrt(ppcm**2 * c_lgt**2 + m3**2 * c_lgt**4)
    E4cm = sqrt(ppcm**2 * c_lgt**2 + m4**2 * c_lgt**4)

    pol3 = atan(ppcm * c_lgt * sin(cm_angle) / (sinh(chi) * E3cm + cosh(chi) * ppcm * c_lgt * cos(cm_angle)))
    pol4 = atan(ppcm * c_lgt * sin(cm_angle) / (sinh(chi) * E4cm + cosh(chi) * ppcm * c_lgt * cos(cm_angle)))

    ejec.energy = T3
    ejec.polar = pol3
    ejec.azimuth = azi

    recoil.energy = T4
    recoil.polar = pol4
    recoil.azimuth = azi + pi

    return ejec, recoil


