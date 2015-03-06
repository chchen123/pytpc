..  currentmodule:: pytpc

Specifying Gases -- `gases`
===========================

This module provides classes and functions for specifying the properties of gases in the detector. This is
accomplished using a generic `Gas` class that can be derived from to make specific gases.

The main purpose of the gas objects is to calculate the amount of energy the tracked particle loses to
ionization during its motion. For the generic `Gas` class, this calculation is done with the
`Bethe equation <https://en.wikipedia.org/wiki/Bethe_formula>`_. Since this formula is neither incredibly
accurate nor numerically stable at low energies, it is better to derive from the `Gas` class and replace this
with a better calculation of the stopping power, perhaps from a fit of experimental data.

Classes
-------

..  autosummary::
    :toctree: generated/

    gases.Gas
    gases.HeliumGas

Functions
---------

..  autosummary::
    :toctree: generated/

    gases.bethe