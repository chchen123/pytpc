""" Constants.py

This module contains a number of constants that are used in the tracking
simulations and elsewhere.

"""

from __future__ import division, print_function
import scipy.constants

e_mc2 = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0]
"""The electron mass in MeV/c**2"""
p_mc2 = scipy.constants.physical_constants['proton mass energy equivalent in MeV'][0]
"""The proton mass in MeV/c**2"""
p_kg = scipy.constants.physical_constants['proton mass'][0]
"""The proton mass in kg"""
e_chg = scipy.constants.physical_constants['elementary charge'][0]
"""The charge of the electron"""
N_avo = scipy.constants.physical_constants['Avogadro constant'][0]
"""Avogadro's number"""
c_lgt = scipy.constants.physical_constants['speed of light in vacuum'][0]
"""The speed of light"""
pi = scipy.constants.pi
"""Pi"""
eps_0 = scipy.constants.physical_constants['electric constant'][0]
"""The electric constant / permittivity of the vacuum"""

pos_step = 1e-3  # in m
num_iters = 100

MeVtokg = 1e6 * e_chg / c_lgt**2
"""Conversion factor for MeV/c^2 to kg"""
amuTokg = 1.66054e-27
"""Conversion factor for amu to kg"""
amuToMeV = 931.494
"""Conversion factor for amu to MeV/c^2"""

degrees = pi / 180.
"""Conversion factor for degrees to radians"""