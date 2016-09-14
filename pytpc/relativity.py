"""Special Relativity Functions

This module contains functions to calculate quantities of interest in special relativity.

"""

from __future__ import division, print_function
import numpy as np
from pytpc.constants import pi, c_lgt
from pytpc.utilities import euler_matrix, constrain_angle
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


def elastic_scatter(proj, target, cm_angle, azi, ret_angles=False):
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
    ret_angles : bool, optional
        Whether to return a tuple of scattering angles as the third result.

    Returns
    -------
    ejec : pytpc.simulation.Particle
        The ejectile particle
    recoil : pytpc.simulation.Particle
        The recoil particle
    (pol3, pol4) : tuple(float)
        The true scattering angles. Only returned if `ret_angles` is True.

    """

    recoil = copy.copy(target)
    ejec = copy.copy(proj)
    m1 = target.mass
    m2 = proj.mass
    m3 = ejec.mass
    m4 = recoil.mass
    T1 = proj.energy  # Note: This is the kinetic energy only

    s = ((m1 + m2)**2 + 2 * m1 * T1)  # The Lorentz invariant. We don't need c's since the mass & mom are in MeV
    pcm = sqrt(((s - m1**2 - m2**2)**2 - 4 * m1**2 * m2**2) / (4 * s))  # COM momentum of the reactants
    ppcm = sqrt(((s - m3**2 - m4**2)**2 - 4 * m3**2 * m4**2) / (4 * s))  # COM momentum of the products

    chi = log((sqrt(pcm**2 + m1**2) + pcm) / m1)  # rapidity of the COM frame

    T3 = sqrt(ppcm**2 + m3**2) * cosh(chi) + ppcm * cos(cm_angle) * sinh(chi) - m3  # ejectile KE
    T4 = sqrt(ppcm**2 + m4**2) * cosh(chi) - ppcm * cos(cm_angle) * sinh(chi) - m4  # recoil KE

    E3cm = sqrt(ppcm**2 + m3**2)  # ejectile COM total energy
    E4cm = sqrt(ppcm**2 + m4**2)  # recoil COM total energy

    pol3 = atan(ppcm * sin(cm_angle) / (sinh(chi) * E3cm + cosh(chi) * ppcm * cos(cm_angle)))  # ejectile lab polar ang
    pol4 = atan(ppcm * sin(cm_angle) / (sinh(chi) * E4cm - cosh(chi) * ppcm * cos(cm_angle)))  # recoil lab polar angle

    ejec.energy = T3
    ejec.polar = pol3
    ejec.azimuth = 0
    ejec.position = proj.position

    recoil.energy = T4
    recoil.polar = pol4
    recoil.azimuth = pi
    recoil.position = proj.position

    # These angles are relative to the projectile's original trajectory, and need to be rotated into the lab frame.
    # This can be done with the Euler angle rotations. We need to use the transpose of the Euler matrix here.
    # (I'm not sure why. It just seems to be what works...)

    eulermat = euler_matrix(phi=proj.azimuth, theta=proj.polar, psi=azi).T  # the transpose is the inverse (ortho. mat.)

    recoil.momentum = eulermat.dot(recoil.momentum)
    ejec.momentum = eulermat.dot(ejec.momentum)

    assert ejec.polar >= 0, 'ejectile polar angle < 0'
    assert recoil.polar >= 0, 'recoil polar angle < 0'

    # The Euler angle rotation might have pushed the azimuthal angles outside [0, 2*pi), so fix them.

    recoil.azimuth = constrain_angle(recoil.azimuth)
    ejec.azimuth = constrain_angle(ejec.azimuth)

    assert 0 <= recoil.azimuth <= 2 * pi, 'recoil azimuth out of bounds'
    assert 0 <= ejec.azimuth <= 2 * pi, 'ejectile azimuth out of bounds'

    assert ejec.energy - T3 < 1e-2, 'ejectile energy was changed by rotation: {} -> {}'.format(T3, ejec.energy)
    assert recoil.energy - T4 < 1e-2, 'recoil energy was changed by rotation: {} -> {}'.format(T4, recoil.energy)

    if ret_angles:
        return ejec, recoil, (pol3, pol4)
    else:
        return ejec, recoil


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
