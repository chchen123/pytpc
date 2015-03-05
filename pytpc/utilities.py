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
    func : function(*args, **kwargs)
        The function to be decorated. Any positional arguments that are non-scalar
        will be converted to an ndarray

    Returns
    -------
    decorated : function(*newargs, **kwargs)
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