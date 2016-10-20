"""Special Relativity Functions

This module contains functions to calculate quantities of interest in special relativity.

"""

from __future__ import division, print_function
import numpy as np
from pytpc.constants import c_lgt, pi
from math import sqrt


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
    vmag = np.linalg.norm(v)
    if vmag >= c_lgt:
        raise ValueError('Velocity was {}, which exceeds c.'.format(vmag))
    return 1 / sqrt(1 - vmag**2 / c_lgt**2)


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
        b = (sqrt(en) * sqrt(en + 2 * mass)) / (en + mass)
    except ZeroDivisionError:
        raise ValueError('mass and energy cannot both be zero')

    if b > 1.0:
        b = 1.0
    return b


def find_proton_params(th3, m1, m2, m3, m4, T):
    s = (m1 + m2)**2 + 2 * m2 * T
    pcm = np.sqrt(((s - m1**2 - m2**2)**2 - 4 * m1**2 * m2**2) / (4 * s))
    ppcm = np.sqrt(((s - m3**2 - m4**2)**2 - 4 * m3**2 * m4**2) / (4 * s))
    chi = np.log((pcm + np.sqrt(m2**2 + pcm**2)) / m2)
    E3cm = np.sqrt(ppcm**2 + m3**2)
#     print(np.sqrt(s), E3cm - m3)

    coshx = np.cosh(chi)
    sinhx = np.sinh(chi)

    root = np.sqrt(coshx**2 * (E3cm**2 + m3**2 * (-coshx**2 + np.cos(th3)**2 * sinhx**2)))
    denom = ppcm * (coshx**2 - np.cos(th3)**2 * sinhx**2)
    sinthcm = np.sin(th3) * (E3cm * np.cos(th3) * sinhx + root) / denom
    p3 = ppcm * sinthcm / np.sin(th3)
    E3 = np.sqrt(p3**2 + m3**2)
    return sinthcm, E3


def find_kine_vert_en(mproj, mtarg, scat_ang, recoil_ke):
    E_recoil = recoil_ke + mtarg
    cm_ang = np.pi - 2 * scat_ang
    p_sq = (mtarg * E_recoil - mtarg**2) / (1 - np.cos(cm_ang))
    sqrts = np.sqrt(mtarg**2 + p_sq) + np.sqrt(mproj**2 + p_sq)
    T = 1 / (2 * mtarg) * (sqrts**2 - (mtarg + mproj)**2)
    return T


# def cm_to_lab_frame(T_recoil_cm, cm_angle, proj_mass, target_mass):
#     en_recoil_cm = T_recoil_cm + target_mass
#     pcm = np.sqrt(en_recoil_cm**2 - target_mass**2)
#     chi = np.log((pcm + np.sqrt(target_mass**2 + pcm**2)) / target_mass)
#
#     en_recoil_lab = en_recoil_cm * np.cosh(chi) + pcm * np.cos(cm_angle) * np.sinh(chi)
#     lab_angle = (pi - cm_angle) / 2  # NOTE: Non-relativistic approximation
#
#     return en_recoil_lab - target_mass, lab_angle


def cm_to_lab_frame(T_recoil_cm, cm_angle, proj_mass, target_mass):
    en_recoil_cm = T_recoil_cm + target_mass
    pcm = np.sqrt(en_recoil_cm**2 - target_mass**2)

    en_target_cm = np.sqrt(pcm**2 + target_mass**2)
    chi = np.log(target_mass / (en_target_cm - pcm))

    en_recoil_lab = en_recoil_cm * np.cosh(chi) - pcm * np.cos(cm_angle) * np.sinh(chi)
    scat_ang = (pi - cm_angle) / 2  # NOTE: Non-relativistic approximation

    return en_recoil_lab - target_mass, scat_ang
