""" Constants.py

This module contains a number of constants that are used in the tracking
simulations and elsewhere.

"""

import scipy.constants

e_mc2 = scipy.constants.physical_constants['electron mass energy equivalent in MeV'][0]
p_mc2 = scipy.constants.physical_constants['proton mass energy equivalent in MeV'][0]
p_kg = scipy.constants.physical_constants['proton mass'][0]
e_chg = scipy.constants.physical_constants['elementary charge'][0]
N_avo = scipy.constants.physical_constants['Avogadro constant'][0]
c_lgt = scipy.constants.physical_constants['speed of light in vacuum'][0]
pi = scipy.constants.pi
eps_0 = scipy.constants.physical_constants['electric constant'][0]

pos_step = 1e-3  # in m
num_iters = 100

MeVtokg = 1e6 * e_chg / c_lgt**2
amuTokg = 1.66054e-27
amuToMeV = 931.494