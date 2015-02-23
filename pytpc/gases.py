from __future__ import division, print_function
import numpy as np

from pytpc.constants import *
import pytpc.relativity as rel
from numpy import log, exp


class Gas(object):
    """Describes a gas in the detector.

    Parameters
    ----------
    molar_mass : number
        Provided in g/mol
    num_electrons : int
        Number of electrons per molecule, or the total Z
    mean_exc_pot : float
        The mean excitation potential, as used in Bethe's formula, in eV
    pressure : float
        The gas pressure in Torr
    """

    def __init__(self, molar_mass, num_electrons, mean_exc_pot, pressure):
        self.molar_mass = molar_mass
        self.num_electrons = num_electrons
        self.mean_exc_pot = mean_exc_pot
        self.pressure = pressure

    @property
    def density(self):
        """Density in g/cm^3"""
        return self.pressure / 760. * self.molar_mass / 24040.

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


class HeliumGas(Gas):
    """Represents the properties of pure helium-4 gas.

    This is a subclass of the `Gas` class for helium-4. The main change is that the energy loss function has been
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
        Gas.__init__(self, 4, 2, 41.8, pressure)

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