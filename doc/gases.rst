..  currentmodule:: pytpc

Gases
=====

The module :mod:`gases` provides some classes to represent gases in the TPC. This is mainly used to calculate the
stopping power of the gas in order to find the energy loss of the tracked particle as it moves through the TPC.

There are two general ways to specify a gas for the program. The first (and best) way is with the
:class:`gases.InterpolatedGas` class. This class calculates the stopping power using an interpolated spline applied to
experimental or externally simulated data. This gives the most precise stopping power data.

The other option is to use the :class:`gases.GenericGas` class, which uses the
`Bethe equation <https://en.wikipedia.org/wiki/Bethe_formula>`_ to calculate the stopping power.
However, since this formula is neither incredibly accurate nor numerically stable at
low energies, this is probably not the best choice.

The abstract :class:`gases.Gas` class is the parent of these two classes, and should not be used directly.

Mixtures
--------

A mixture of gases can be created using the :class:`gases.InterpolatedGasMixture` class. This uses the same method
as the :class:`gases.InterpolatedGas` class to calculate the stopping power of each component gas. The stopping powers are
then averaged together using the partial density of each gas as a weighting factor.

Usage
-----

The :class:`gases.InterpolatedGas` class can be loaded by specifying a gas name and pressure::

    gas = pytpc.gases.InterpolatedGas('helium', 150.)  # helium gas at 150 torr, using the interpolated class

The name of the gas must match a file in the directory :file:`$PYTPC/data/gases`, where :file:`$PYTPC` is the directory
where the ``pytpc`` module is installed. Some example gas names are ``'helium'`` and ``'carbon dioxide'``.

The generic class can be instantiated with the molar mass, number of electrons per molecule, mean excitation potential,
and pressure of the gas::

    gas = pytpc.gases.GenericGas(4, 100., 2, 41.8)  # helium gas at 100 torr, using the generic class

To make a mixture of gases, specify the total pressure and the composition of the gas::

    # This makes a mixture of 90% helium and 10% CO2
    mix = pytpc.gases.InterpolatedGasMixture(150., ('helium', 0.9), ('carbon dioxide', 0.1))

Any mixture of gases can be specified this way. The only restrictions are that each component must be a valid
:class:`InterpolatedGas` and the proportions of the gases must sum to 1.

Format of gas data files
------------------------

The files in the gas data directory mentioned above are JSON files with the following keys:

.. cssclass:: table-striped

============  ============================================
     Key                       Value
============  ============================================
'molar_mass'  The molar mass of the gas
'dedx'        The energy loss information, in MeV/(g/cm^2)
============  ============================================

Any other keys will be ignored.

The data itself could come from a paper, from SRIM, or from a website like http://www.nist.gov/pml/data/star/.

In the JSON format, a file might look like this:

..  code-block:: json

    {
        "molar_mass": 4.002,

        "dedx": [[0.001, 382.2],
                 [0.0015, 367.8],
                 [0.002, 360.3],
                 ...
                 ]
    }

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
    GenericGas
    InterpolatedGas
    InterpolatedGasMixture

..  rubric:: Functions

..  autosummary::
    :toctree: generated/

    bethe