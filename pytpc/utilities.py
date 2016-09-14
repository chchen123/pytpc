"""
Utilities
=========

Some common functions that could be useful in many places.

"""

import numpy as np
from numpy import sin, cos
from functools import wraps


def numpyize(func):
    """Decorator that converts all array-like arguments to NumPy ndarrays.

    Parameters
    ----------
    func : function
        The function to be decorated. Any positional arguments that are non-scalar
        will be converted to an ndarray

    Returns
    -------
    decorated : function
        The decorated function, with all array-like arguments being ndarrays

    """
    @wraps(func)
    def decorated(*args, **kwargs):
        newargs = list(args)
        for i, a in enumerate(newargs):
            if not np.isscalar(a):
                newargs[i] = np.asanyarray(a)
        return func(*newargs, **kwargs)

    return decorated


def skew_matrix(angle):
    """Creates a left/right skew transformation matrix.

    Parameters
    ----------
    angle : float
        The angle to skew by, in radians

    Returns
    -------
    mat : ndarray
        The transformation matrix
    """

    mat = np.array([[1, 1/np.tan(angle)],
                    [0, 1]])
    return mat


def rot_matrix(angle):
    """Creates a two-dimensional rotation matrix.

    Parameters
    ----------
    angle : float
        The angle to rotate by, in radians

    Returns
    -------
    mat : ndarray
        The transformation matrix
    """

    mat = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    return mat


def tilt_matrix(angle):
    """Creates a 3D rotation matrix appropriate for the detector's tilt angle.

    This corresponds to a rotation about the x axis of ``-angle`` radians.

    Parameters
    ----------
    angle : float
        The angle to rotate by, in radians. Note that the matrix will be for a rotation of **negative** this angle.

    Returns
    -------
    mat : ndarray
        The rotation matrix

    """

    mat = np.array([[1, 0, 0],
                    [0, np.cos(-angle), -np.sin(-angle)],
                    [0, np.sin(-angle), np.cos(-angle)]])
    return mat


def euler_matrix(phi, theta, psi):
    """Return the Euler matrix for a three-dimensional rotation through the given angles.

    The particular rotation scheme used here is z-y-z, or the following operations:

        1) A rotation of phi about the z axis
        2) A rotation of theta about the new y axis
        3) A rotation of psi about the new z axis

    The source matrices for each operation use the convention found in Goldstein [1]_, which appears to be
    a negative rotation (or perhaps a passive transformation?).

    Parameters
    ----------
    phi : number
        The angle for the first rotation, about z, in radians.
    theta : number
        The angle for the second rotation, about y, in radians.
    psi : number
        The angle for the third rotation, about z, in radians.

    Returns
    -------
    mat : ndarray
        The Euler angle rotation matrix.

    References
    ----------
    .. [1] Goldstein, H., Poole, C., and Safko, J. (2002). *Classical mechanics*, 3rd ed.
       Addison Wesley, San Francisco, CA. Pg. 153.
    """

    mat = np.array([[cos(theta)*cos(phi)*cos(psi) - sin(phi)*sin(psi),
                     cos(theta)*cos(psi)*sin(phi) + cos(phi)*sin(psi),
                     -(cos(psi)*sin(theta))],
                    [-(cos(psi)*sin(phi)) - cos(theta)*cos(phi)*sin(psi),
                     cos(phi)*cos(psi) - cos(theta)*sin(phi)*sin(psi),
                     sin(theta)*sin(psi)],
                    [cos(phi)*sin(theta), sin(theta)*sin(phi), cos(theta)]])

    return mat


def constrain_angle(x):
    """Constrain the given angle to lie between 0 and 2*pi.

    Angles outside the bounds are rotated by 2*pi until they lie within the bounds.

    Parameters
    ----------
    x : number
        The angle to constrain, in radians.

    Returns
    -------
    y : number
        The constrained angle, which will lie between 0 and 2*pi.
    """
    y = x % (2*np.pi)
    if y < 0:
        y += 2*np.pi

    return y


def create_fields(emag, bmag, tilt):
    """Convenience function to create electric and magnetic field vectors.

    Parameters
    ----------
    emag : number
        The electric field magnitude, in SI units.
    bmag : number
        The magnetic field magnitude, in Tesla.
    tilt : number
        The detector tilt angle, in radians.

    Returns
    -------
    ef : ndarray
        The electric field
    bf : ndarray
        The magnetic field

    """

    ef = np.array([0, 0, emag])

    bf_orig = np.array([0, 0, bmag])
    trans = tilt_matrix(tilt)
    bf = trans.dot(bf_orig)

    return ef, bf


def find_vertex_energy(beam_intercept, beam_enu0, beam_mass, beam_chg, gas):
    ei = beam_enu0 * beam_mass
    ri = gas.range(ei, beam_mass, beam_chg)  # this is in meters
    rf = ri - (1.0 - beam_intercept)
    ef = gas.inverse_range(rf, beam_mass, beam_chg)
    return ef
