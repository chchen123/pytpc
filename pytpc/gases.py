from __future__ import division, print_function
import numpy as np

from pytpc.constants import *
import pytpc.relativity as rel
from numpy import log, exp
from scipy.interpolate import InterpolatedUnivariateSpline

import os
import json


class Gas(object):
    """Base class that describes a gas in the detector.

    This should generally not be used directly as it doesn't provide an energy loss information. Instead use one of
    its derived classes.

    Parameters
    ----------
    molar_mass : number
        Provided in g/mol
    pressure : float
        The gas pressure in Torr

    Attributes
    ----------
    density : float
        The density of the gas
    molar_mass : float
        The molar mass of the gas molecules
    pressure : float
        The gas pressure

    See Also
    --------
    GenericGas
    InterpolatingGas
    InterpolatingGasMixture
    """

    def __init__(self, molar_mass, pressure):
        self.molar_mass = molar_mass
        self.pressure = pressure

    @property
    def density(self):
        """Density in g/cm^3"""
        return self.pressure / 760. * self.molar_mass / 24040.

    def energy_loss(self, en, proj_mass, proj_charge):
        raise NotImplementedError()


class InterpolatedGas(Gas):
    r"""A gas that calculates its stopping power by interpolating from experimental data.

    The experimental data is stored in the directory ``PYTPCROOT/data/gases`` where ``PYTPCROOT`` is wherever the
    whole package is stored.

    On instantiation, this will look for a JSON file corresponding to the name given to __init__. The JSON file
    must have (at least) the following keys:

    ============  =========================================================
         Key                                Value
    ============  =========================================================
    'molar_mass'  The molar mass as a float
    'dedx'        The energy loss, as a list of pairs like [energy, dedx].
                  The energy loss is assumed to be in units of MeV/(g.cm^2)
    ============  =========================================================

    After reading in the data, it is interpolated using SciPy's ``InterpolatedUnivariateSpline`` class. This spline
    is then used by the `energy_loss` function to produce the energy loss.

    Parameters
    ----------
    name : string
        The name of the gas. This must match an existing file in the gas data folder.
    pressure : float
        The gas pressure in Torr

    Attributes
    ----------
    density : float
        The density of the gas
    molar_mass : float
        The molar mass of the gas molecules
    pressure : float
        The gas pressure
    """

    def __init__(self, name, pressure):

        # TODO: Make this less of a kludge
        filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'gases', name + '.json'))
        try:
            with open(filename) as f:
                gas_dict = json.load(f)

        except FileNotFoundError:
            raise ValueError('Gas data file not found. Is {} a valid gas name?'.format(name))

        dedx = np.array(gas_dict['dedx'])
        self.spline = InterpolatedUnivariateSpline(dedx[:, 0], dedx[:, 1])

        Gas.__init__(self, gas_dict['molar_mass'], pressure)

    def energy_loss(self, en, proj_mass, proj_charge):
        """Calculates the energy loss of a projectile in the gas using the interpolated spline.

        Parameters
        ----------
        en : float
            The projectile's kinetic energy in MeV
        proj_mass : int
            The mass number of the projectile
        proj_charge : int
            The charge number of the projectile

        Returns
        -------
        float
            The stopping power of the gas, in MeV/m

        """

        if proj_mass != 4 or proj_charge != 2:
            raise NotImplementedError('Not implemented for particles other than alpha')

        return self.spline(en) * self.density * 100  # in MeV/m


class GenericGas(Gas):

    def __init__(self, molar_mass, pressure, num_electrons, mean_exc_pot):
        self.num_electrons = num_electrons
        self.mean_exc_pot = mean_exc_pot
        Gas.__init__(self, molar_mass, pressure)

    @property
    def electron_density(self):
        """Electron density per cm^3"""
        return N_avo * self.num_electrons * self.density / self.molar_mass

    @property
    def electron_density_per_m3(self):
        """Electron density per m^3"""
        return self.electron_density * 1e6

    def energy_loss(self, en, proj_mass, proj_charge):
        """Finds the energy loss of a projectile in the gas.

        This default version uses the Bethe formula (defined by the function `bethe`) to calculate the energy loss.
        This function may be overridden with an empirical fit to give better results.

        Parameters
        ----------
        en : float
            The kinetic energy of the projectile, in MeV
        proj_mass : int
            The mass number of the projectile
        proj_charge : int
            The charge number of the projectile

        Returns
        -------
        dedx : float
            The stopping power in MeV/m
        """
        beta = rel.beta(en, proj_mass * p_mc2)
        return bethe(beta, proj_charge, self.electron_density_per_m3, self.mean_exc_pot)


class InterpolatedGasMixture(Gas):

    def __init__(self, pressure, *args):

        self.components = []
        molar_mass = 0.0
        total_prop = 0.0
        for comp in args:
            assert len(comp) == 2, 'specify components as (gas name, proportion)'
            name, prop = comp
            assert 0 < prop <= 1, 'gas proportions should be between 0 and 1'
            gas = InterpolatedGas(name, pressure * prop)
            molar_mass += prop * gas.molar_mass
            total_prop += prop
            self.components.append((gas, prop))

        assert total_prop == 1.0, 'total gas fractions must sum to 1.0'

        Gas.__init__(self, molar_mass, pressure)

    def energy_loss(self, en, proj_mass, proj_charge):
        """Calculates the energy loss of a projectile in the gas using the interpolated spline.

        Parameters
        ----------
        en : float
            The projectile's kinetic energy in MeV
        proj_mass : int
            The mass number of the projectile
        proj_charge : int
            The charge number of the projectile

        Returns
        -------
        float
            The stopping power of the gas, in MeV/m

        """

        if proj_mass != 4 or proj_charge != 2:
            raise NotImplementedError('Not implemented yet for particles other than alphas')

        # The total energy loss is the sum of the energy lost to each component.
        # This is valid since each component has the correct partial pressure.
        de = sum([g.energy_loss(en, 4, 2) for g, prop in self.components])
        return de  # in MeV/m


class HeliumGas(GenericGas):
    """Represents the properties of pure helium-4 gas.

    This is a subclass of the `GenericGas` class for helium-4. The main change is that the energy loss function has been
    overridden with an empirical fit. The fit data comes from the AT-TPC Fortran simulation, and the data itself may
    have originally come from Northcliffe and Schilling (1970).

    Parameters
    ----------
    pressure : float
        The gas pressure, in Torr

    See Also
    --------
    Gas

    """

    def __init__(self, pressure):
        GenericGas.__init__(self, 4, pressure, 2, 41.8)

    def energy_loss(self, en, proj_mass, proj_charge):
        """Calculates the energy loss of a projectile in the gas.

        This is an empirical fit of experimental data, and only works for a small variety of projectiles.

        The currently supported projectiles are:

        - Protons
        - Alphas / 4He

        Parameters
        ----------
        en : float
            The projectile's kinetic energy in MeV
        proj_mass : int
            The mass number of the projectile
        proj_charge : int
            The charge number of the projectile

        Returns
        -------
        float
            The stopping power of the gas, in MeV/m

        Raises
        ------
        ValueError
            If the projectile is one of the known projectiles. See the list above for acceptible projectiles.

        """

        if proj_mass == 1 and proj_charge == 1:
            # Protons in helium gas
            result = 6.5*(1./en**0.83)*(1./(20.+1.6/(en**1.3))) + 0.2*exp(-30.*(en-0.1)**2)

        elif proj_mass == 4 and proj_charge == 2:
            # Helium ions in helium gas. Only good down to K.E. = 10 keV
            result = 10.*(1./en**0.83)*(1./(2.5+1.6/(en**.5))) + 0.05*exp(-(en-0.5)**2)

        else:
            raise ValueError('Unknown projectile: mass={m}, charge={q}'.format(m=proj_mass,
                                                                               q=proj_charge))

        # Result is in MeV / (mg.cm^2), so convert and mutliply by density
        result *= 1000 * self.density * 100  # This is in MeV/m
        return result


class HeCO2Gas(GenericGas):
    """Represents a mixture of 90 percent helium and 10 percent carbon dioxide.

    This is a subclass of the `GenericGas` class with a replacement for the energy loss function.
    The energy loss function
    used here is a fit to data from the NIST ASTAR website (http://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html).

    ..  Warning::
        Some of the properties inherited from the generic `Gas` class (such as electron density properties) are
        not valid for this gas. They will return a value, but it will not be correct.

    Parameters
    ----------
    pressure : float
        The gas pressure, in Torr

    See Also
    --------
    Gas

    """

    def __init__(self, pressure):
        he_mol_mass = 4.002  # g/mol
        co2_mol_mass = 44.01  # g/mol
        mol_mass = he_mol_mass * 0.9 + co2_mol_mass * 0.1
        GenericGas.__init__(self, mol_mass, pressure, 2, 41.8)  # these parameters are wrong, but they don't matter (I hope)

    @staticmethod
    def _fit_func(en, a, b, c, d, e, f, g, h):
        """The fit function for the energy loss parameterization.

        This is in the form required for scipy.optimize.curve_fit.

        All of the parameters after the first one are coefficients.

        Parameters
        ----------
        en : number
            The energy

        Returns
        -------
        number
            The energy loss
        """
        return a*(1./en**b)*(1./(c+d/(en**e))) + f*exp(-g*(en-h)**2)

    def energy_loss(self, en, proj_mass, proj_charge):
        """Calculates the energy loss of a projectile in the gas.

        This is an empirical fit of experimental data, and only works for a small variety of projectiles.

        The currently supported projectiles are:

        - Alphas / 4He

        Parameters
        ----------
        en : float
            The projectile's kinetic energy in MeV
        proj_mass : int
            The mass number of the projectile
        proj_charge : int
            The charge number of the projectile

        Returns
        -------
        float
            The stopping power of the gas, in MeV/m

        Raises
        ------
        ValueError
            If the projectile is one of the known projectiles. See the list above for acceptible projectiles.

        """

        if proj_mass == 4 and proj_charge == 2:
            # This is an alpha particle
            fit_params = np.array([3.96952385e+02,   9.33364832e-01,   9.59137201e-02,
                                   8.82262274e-02,   1.51501228e+00,  -1.82205350e+03,
                                   9.93911292e+03,  -1.81747643e-01])
            result = self._fit_func(en, *fit_params)

        else:
            raise ValueError('Unknown projectile: mass={m}, charge={q}'.format(m=proj_mass,
                                                                               q=proj_charge))

        # Result of fit function is in MeV / (g/cm^2), so multipy by density
        result *= self.density * 100  # This is in MeV/m
        return result


def bethe(beta, z, ne, exc_en):
    """ Find the stopping power of the gas.

    Parameters
    ----------
    beta : float
        The beta of the projectile
    z : int
        The charge number of the projectile
    ne : float
        The electron density of the gas, in m^-3
    exc_en : float
        The excitation energy of the gas, in eV

    Returns
    -------
    dedx : float
        The stopping power in MeV/m

    """
    exc_en *= 1e-6  # convert to MeV
    beta_sq = beta**2

    if beta_sq == 0.0:
        # The particle has stopped, so the dedx should be infinite
        dedx = float('inf')
    elif beta_sq == 1.0:
        # This is odd, but then I guess dedx -> 0
        dedx = 0
    else:
        frnt = ne * z**2 * e_chg**4 / (e_mc2 * MeVtokg * c_lgt**2 * beta_sq * 4 * pi * eps_0**2)
        lnt = log(2 * e_mc2 * beta_sq / (exc_en * (1 - beta_sq)))
        dedx = frnt*(lnt - beta_sq)  # this should be in SI units, J/m

    return dedx / e_chg * 1e-6  # converted to MeV/m