"""Special Relativity Functions

This module contains functions to calculate quantities of interest in special relativity.

"""

import numpy
from .constants import *
from math import sqrt


def gamma(v):
    """Returns gamma.

    Gamma is defined as $$\gamma = \frac{1}{\sqrt{1-\beta^2}}$$ where $\beta = \frac{v}{c}$.

    The argument v may be a number or an array-like object.
    """
    vmag = numpy.linalg.norm(v)
    if vmag >= c_lgt:
        raise ValueError('Velocity was {}, which exceeds c.'.format(vmag))
    return 1/sqrt(1-vmag**2/c_lgt**2)


def beta(en, mass):
    """ Returns beta, or v / c.

    The arguments should be in compatible units.

    Arguments
    ---------
    en : the relativistic kinetic energy
    mass : the rest mass
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
