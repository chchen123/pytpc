..  currentmodule:: pytpc

Utilities and Other Functionality
=================================

This page describes some of the other functionality that's provided by PyTPC.

Working with the Pad Plane
--------------------------

The :mod:`padplane` module can be used to describe the pad plane of the Micromegas.

..  rubric:: Functions

..  autosummary::
    :toctree: generated/

    padplane.generate_pad_plane

..  rubric:: Attributes

..  autosummary::
    :toctree: generated/

    padplane.pad_height
    padplane.pad_base
    padplane.inner_pad_height
    padplane.inner_pad_base
    padplane.xgap
    padplane.ygap
    padplane.tri_height
    padplane.tri_base
    padplane.inner_tri_height
    padplane.inner_tri_base

Special Relativity
------------------

The :mod:`relativity` module contains some functions that are needed for relativistic calculations.

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    relativity.beta
    relativity.gamma

Constants
---------

The :mod:`constants` provides a number of physical constants.

..  rubric:: Physical constants

..  autosummary::
    :toctree: generated/

    constants.e_mc2
    constants.p_mc2
    constants.p_kg
    constants.e_chg
    constants.N_avo
    constants.c_lgt
    constants.pi
    constants.eps_0

..  rubric:: Conversion factors

..  autosummary::
    :toctree: generated/

    constants.MeVtokg
    constants.amuTokg
    constants.amuToMeV
    constants.degrees

Utilities
---------

The :mod:`utilities` module contains some other utilities of general usefulness.

..  rubric:: Decorators

..  autosummary::
    :toctree: generated/

    utilities.numpyize

..  rubric:: Transformation matrices

..  autosummary::
    :toctree: generated/

    utilities.rot_matrix
    utilities.skew_matrix


