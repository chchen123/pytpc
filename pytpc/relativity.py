"""Special Relativity Functions

This module contains functions to calculate quantities of interest in special relativity.

"""

from __future__ import division, print_function
import numpy
from pytpc.constants import *
from math import sqrt


def gamma(v):
    r"""Returns gamma.

    Gamma is defined as :math:`\gamma = \frac{1}{\sqrt{1-\beta^2}}` where :math:`\beta = \frac{v}{c}`.

    **Arguments**

    v : array-like or number
        the velocity
    """
    vmag = numpy.linalg.norm(v)
    if vmag >= c_lgt:
        raise ValueError('Velocity was {}, which exceeds c.'.format(vmag))
    return 1/sqrt(1-vmag**2/c_lgt**2)


def beta(en, mass):
    """ Returns beta, or :math:`v / c`.

    The arguments should be in compatible units.

    **Arguments**

    en : int or float
        the relativistic kinetic energy
    mass : int or float
        the rest mass

    :param int en: the energy
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
