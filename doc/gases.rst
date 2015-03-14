..  currentmodule:: pytpc

Gases
=====

The module :mod:`gases` provides some classes to represent gases in the TPC. This is mainly used to calculate the
stopping power of the gas in order to find the energy loss of the tracked particle as it moves through the TPC.

The generic :class:`gases.Gas` class uses the `Bethe equation <https://en.wikipedia.org/wiki/Bethe_formula>`_ to
calculate the stopping power. However, since this formula is neither incredibly accurate nor numerically stable at
low energies, it is better to derive from the `Gas` class and replace this with a better calculation of the stopping
power, perhaps from a fit of experimental data.

Usage
-----

The generic class can be instantiated with the molar mass, number of electrons per molecule, mean excitation potential,
and pressure of the gas::

    gas = pytpc.gases.Gas(4, 2, 41.8, 100.)  # helium gas at 100 torr, using the generic class

The specific gases are a bit simpler to use, as all that needs to be specified is the pressure::

    he = pytpc.gases.HeliumGas(100.)  # helium gas at 100 torr, using the specific class

Currently Implemented Gases
---------------------------

The following gases are currently available:

- Pure helium-4
- Generic

Deriving a New Gas
------------------

A new gas can be derived by adding a subclass to the :file:`gases.py` file. A basic outline of a gas is below::

    class SomeNewGas(Gas):

        def __init__(self, pressure):
            # Call the superclass to initialize it. Pass in the parameters of the gas.
            Gas.__init__(self, 4, 2, 41.8, pressure)

        def energy_loss(self, en, proj_mass, proj_charge):
            # This overrides the Bethe function in the generic gas.
            # Insert a fit from data or some other code here to calculate the
            # stopping power.
            pass

Naturally, this should also be filled in with appropriate docstrings to add to this documentation!

API Reference
-------------

pytpc.gases
~~~~~~~~~~~

This module describes the gases used in the detector.

..  currentmodule:: pytpc.gases

..  rubric:: Classes

..  autosummary::
    :toctree: generated/

    Gas
    HeliumGas

..  rubric:: Functions

..  autosummary::
    :toctree: generated/

    bethe