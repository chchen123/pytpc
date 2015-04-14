"""
Utilities
=========

Some common functions that could be useful in many places.

"""

import numpy as np
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