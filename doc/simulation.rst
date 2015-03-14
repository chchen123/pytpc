..  currentmodule:: pytpc

Physics Model and Simulation of Tracks
======================================

This module establishes the physics model used by the Kalman filter, and can also be used to produce simulated particle
tracks for testing.

Model Used
----------

The Kalman filter requires a physics model to predict the next position of the tracked particle at each step. The model
used here is fairly basic, but seems to work reasonably well.

Each prediction consists of two basic steps:

1. Move the particle forward using the Lorentz force.
2. Adjust the energy using a model of the gas stopping power.

The implementation of this is, naturally, a bit more complex.

Particles
---------

A particle is represented in code by the :class:`simulation.Particle` class (also available in the :mod:`pytpc`
namespace). This is done purely for computational convenience. Storing particle attributes in objects provides a
convenient way to calculate things like momentum, energy, and velocity without having to have tons of conversion
functions.

A particle can be created by specifying, at a minimum, its mass and charge numbers::

    pt = pytpc.Particle(4, 2)  # an alpha particle

The particle also has attributes representing its momentum, energy, velocity, gamma, beta, mass in various units, polar
and azimuthal angles of motion, and state vector. That last option is most useful to the Kalman filter.

Since most of these attributes are implemented as Python properties, setting one of them is sufficient to completely
update the rest.

Fields
------

The electric and magnetic fields should be specified with NumPy arrays.

..  warning::
    Be careful with the coordinate systems when analyzing data from the tilted AT-TPC. It's easiest to use the
    coordinates of the TPC, and rotate the magnetic field accordingly.

Simulating a Track
------------------

A track can be simulated with the :func:`simulation.track` function. This takes the electric and magnetic fields, the
particle, and the gas in the detector as arguments::

    pt = pytpc.Particle(4, 2, 2.)     # an alpha particle with 2.0 MeV/u
    he = pytpc.gases.HeliumGas(100.)  # helium gas at 100 torr
    efield = np.array([0, 0, 15e3])   # the electric field
    bfield = np.array([0, 0, -1])     # the magnetic field

    res = pytpc.track(pt, he, efield, bfield)

The results are a dictionary containing keys for position, momentum, energy, and more.

..  note::
    The gases are described in the :mod:`gases` module.

API Reference
-------------

pytpc.simulation
~~~~~~~~~~~~~~~~

This module contains functionality for simulating tracks and modeling the physics for the Kalman filter.

..  currentmodule:: pytpc.simulation

..  rubric:: Classes

..  autosummary::
    :toctree: generated/

    Particle

..  rubric:: Functions

..  autosummary::
    :toctree: generated/

    track
    find_next_state
    lorentz
    threshold
    drift_velocity_vector