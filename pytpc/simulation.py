"""
simulation.py
=============

Contains code for simulating the tracking of particles in a TPC.

"""

from __future__ import division, print_function
import numpy as np
from math import sin, cos


def lorentz(vel, ef, bf, charge):
    """Calculate the Lorentz (electromagnetic) force.

    This is defined as

    ..  math::
        \\mathbf{F} = q(\\mathbf{E} + \\mathbf{v} \\times \\mathbf{B})

    ..  note::
        The units of the inputs must be consistant to get a consistant result.

    Parameters
    ----------
    vel : array-like
        The velocity
    ef : array-like
        The electric field
    bf : array-like
        The magnetic field
    charge : float or int
        The charge of the particle

    Returns
    -------
    force : array-like
        The electromagnetic force, in whatever units the inputs generate
    """

    vx, vy, vz = vel
    ex, ey, ez = ef
    bx, by, bz = bf

    fx = charge * (ex + vy * bz - vz * by)
    fy = charge * (ey + vz * bx - vx * bz)
    fz = charge * (ez + vx * by - vy * bx)

    return np.array([fx, fy, fz])


def threshold(value, threshmin=0.):
    """Applies a threshold to the given value.

    Any value less than threshmin will be replaced by threshmin.

    Parameters
    ----------
    value : number
        The value to be thresholded
    threshmin : number
        The minimum threshold value

    """
    if value < threshmin:
        return threshmin
    else:
        return value


def drift_velocity_vector(vd, efield, bfield, tilt):
    r"""Returns the drift velocity vector.

    This includes the effects due to the Lorentz angle from the non-parallel electric and magnetic fields.

    Parameters
    ----------
    vd : number
        The drift velocity, in cm/us.
    efield : number
        The electric field magnitude, in SI units.
    bfield : number
        The magnetic field magnitude, in Tesla.
    tilt : number
        The angle between the electric and magnetic fields, in radians.

    Returns
    -------
    ndarray
        The drift velocity vector

    Notes
    -----
    This formula is a steady-state solution to the Lorentz force equation with a damping term:

    ..  math:: m \dot{\mathbf{v}} = q(\mathbf{E} + \mathbf{v}\times\mathbf{B}) - \beta \mathbf{v}

    The form of the damping term is not important since I'm not actually deriving the solution here. References [1]_
    state that the solution is

    .. math::
        \mathbf{v}_D = \frac{\mu E}{1 + \omega^2\tau^2} [ \hat E + \omega\tau \hat{E}\times\hat{B}
            + \omega^2\tau^2(\hat{E}\cdot\hat{B})\hat{B} ]

    If we let

    .. math::
        \omega = \frac{eB}{m}; \quad \tau = \frac{m v_D}{eE}; \quad \mu = \frac{e\tau}{m}

    and assume that the electric field is along z and the magnetic field is tilted away from z at an angle `tilt` along
    the y direction, then this reduces to the form implemented here.

    References
    ----------
    ..  [1] T. Lohse and W. Witzeling, in *Instrumentation in High Energy Physics*, edited by F. Sauli
        (World Scientific, Singapore, 1992), p. 106.

    """

    ot = bfield / efield * vd * 1e4  # omega * tau in formula. The numerical factor converts cm/us -> m/s

    front = vd / (1 + ot**2)

    xcomp = -front * ot * sin(tilt)
    ycomp = front * ot**2 * cos(tilt) * sin(tilt)
    zcomp = front * (1 + ot**2 * cos(tilt)**2)

    return np.array([xcomp, ycomp, zcomp])  # in cm/us, as long as vd had the right units
