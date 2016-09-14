"""Particle Tracking

This module contains code to track a particle in data.

"""

from __future__ import division, print_function
import numpy as np
from functools import lru_cache

from pytpc.padplane import generate_pad_plane


pads = generate_pad_plane()  # FIXME: Take rotation angle
pr = np.round(pads)


@lru_cache(maxsize=10240)
def find_adj(p, depth=0):
    """Recursively finds the neighboring pads to pad `p`.

    This function returns a set of neighboring pad numbers. The neighboring pads are determined by looking for
    all pads that (approximately) share a vertex with the given pad.

    .. note::
        Since only vertices are compared, this algorithm will not identify a neighboring pad that only shares a side,
        and not a vertex.

    The `depth` parameter allows recursion an arbitrary number of times. For instance, if `depth == 1`, the function
    will return the pads adjacent to pad `p` and those adjacent to its immediate neighbors.

    For efficiency, the results of this function are cached.

    Parameters
    ----------
    p : int
        The pad number whose neighbors you want to find
    depth : int
        The number of times to recurse

    Returns
    -------
    adj : set
        The adjacent pads

    """
    adj = set()
    for pt in pr[p]:
        i = np.argwhere(np.any(np.all(np.abs(pr[:, :] - pt) < np.array([2, 2]), axis=-1), axis=-1))
        adj |= set(i.ravel())

    if depth > 0:
        for a in adj.copy():
            adj |= find_adj(a, depth - 1)

    return adj
