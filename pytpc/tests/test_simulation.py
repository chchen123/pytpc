"""Unit tests for simulation.py"""

from __future__ import division, print_function
import unittest
import pytpc.simulation as sim
import pytpc.gases
from math import sqrt, sin, cos, atan2
import numpy
import numpy.testing as nptest
from pytpc.constants import *


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.en_i = 3
        self.Z_i = 2
        self.A_i = 4
        self.pos_i = (1., 1., 1.)
        self.azi_i = pi/2
        self.pol_i = pi/3
        self.p = sim.Particle(self.A_i, self.Z_i, self.en_i, self.pos_i, self.azi_i, self.pol_i)

    def test_mass(self):
        exp_mass = self.p.mass_num * p_mc2
        self.assertEqual(self.p._mass, exp_mass)
        self.assertEqual(self.p.mass, exp_mass)
        self.assertEqual(self.p.mass_kg, exp_mass * MeVtokg)

    def test_energy(self):
        exp_energy = self.en_i * self.A_i
        self.assertEqual(self.p.energy, exp_energy)
        self.assertEqual(self.p.energy_per_particle, exp_energy / self.A_i)
        self.assertEqual(self.p.energy_j, exp_energy * 1e6 * e_chg)

    def test_energy_setter(self):
        new_en = (self.en_i + 1) * self.A_i
        self.p.energy = new_en
        self.assertEqual(self.p.energy, new_en)

    def test_energy_per_particle_setter(self):
        new_en_per_u = (self.en_i / self.A_i) + 1
        self.p.energy_per_particle = new_en_per_u
        self.assertEqual(self.p.energy / self.p.mass_num, new_en_per_u)

    def test_gamma(self):
        exp_gamma = self.p.energy / self.p.mass + 1
        self.assertAlmostEqual(self.p.gamma, exp_gamma, places=6)

    def test_beta(self):
        exp_gamma = self.p.energy / self.p.mass + 1
        exp_beta = sqrt(1 - 1/exp_gamma**2)
        self.assertAlmostEqual(self.p.beta, exp_beta, places=6)

    def test_velocity(self):
        beta = self.p.beta
        exp_vel = numpy.array([beta*c_lgt*cos(self.azi_i)*sin(self.pol_i),
                               beta*c_lgt*sin(self.azi_i)*sin(self.pol_i),
                               beta*c_lgt*cos(self.pol_i)])
        nptest.assert_allclose(self.p.velocity, exp_vel)

    def test_velocity_setter(self):
        new_vel = 2 * self.p.velocity
        vx, vy, vz = new_vel
        self.p.velocity *= 2
        nptest.assert_allclose(self.p.velocity, new_vel)

        exp_azi = atan2(vy, vx)
        exp_pol = atan2(sqrt(vx**2 + vy**2), vz)
        self.assertAlmostEqual(self.p.azimuth, exp_azi, places=6)
        self.assertAlmostEqual(self.p.polar, exp_pol, places=6)

        exp_en = (1/sqrt(1 - numpy.linalg.norm(new_vel)**2 / c_lgt**2) - 1)*self.p.mass
        self.assertAlmostEqual(self.p.energy, exp_en, places=6)

    def test_momentum(self):
        mom = self.p.momentum
        self.assertEqual(len(mom), 3, msg='momentum is scalar')

        exp_mag = sqrt((self.p.energy + self.p.mass)**2 - self.p.mass**2)
        self.assertAlmostEqual(numpy.linalg.norm(mom), exp_mag, places=6)

    def test_momentum_zero(self):
        self.p.energy = 0
        nptest.assert_equal(self.p.momentum, 0)

    def test_state_vector(self):
        sv = self.p.state_vector
        nptest.assert_equal(sv[0:3], self.p.position)
        nptest.assert_equal(sv[3:6], self.p.momentum)

    def test_state_vector_setter(self):
        new_pos = numpy.array((4, 2, 1))
        new_mom = self.p.momentum / 2
        self.p.state_vector = numpy.hstack((new_pos, new_mom))

        nptest.assert_allclose(self.p.position, new_pos)
        nptest.assert_allclose(self.p.momentum, new_mom)

    def test_momentum_si(self):
        mom = self.p.momentum
        exp = mom * 5.344286e-22  # conversion factor for MeV/c -> kg.m/s from Wolfram Alpha
        nptest.assert_allclose(exp, self.p.momentum_si)

    def test_momentum_si_setter_nonrel(self):
        vel = numpy.array((5., 4., 3.8))
        mom = self.p.mass_kg * vel
        self.p.momentum_si = mom
        nptest.assert_allclose(self.p.velocity, vel)

    def test_azimuth_setter(self):
        self.p.azimuth = pi/24
        self.assertAlmostEqual(self.p.azimuth, pi/24, delta=1e-3)

    def test_polar_setter(self):
        self.p.polar = pi/24
        self.assertAlmostEqual(self.p.polar, pi/24, delta=1e-3)


class TestLorentz(unittest.TestCase):
    """Tests for sim.lorentz function"""

    def do_test_values(self, vel=numpy.array((3e6, 4e4, 1e2)), bf=numpy.array((0, 0, -2)),
                       ef=numpy.array((0, 0, 1e6)), charge=4*e_chg):
        """A helper function to run tests when the goal is comparing the return value."""
        res = sim.lorentz(vel, ef, bf, charge)
        exp = charge*(ef + numpy.cross(vel, bf))
        nptest.assert_allclose(res, exp)

    def test_zero_charge(self):
        self.do_test_values(charge=0)

    def test_zero_efield(self):
        self.do_test_values(ef=numpy.zeros(3))

    def test_zero_bfield(self):
        self.do_test_values(bf=numpy.zeros(3))

    def test_zero_ef_bf(self):
        self.do_test_values(ef=numpy.zeros(3), bf=numpy.zeros(3))

    def test_zero_vel(self):
        self.do_test_values(vel=numpy.zeros(3))

    def test_scalar_vel(self):
        self.assertRaises(TypeError, self.do_test_values, vel=1, bf=1)


class TestFindNextState(unittest.TestCase):
    """Tests for sim.find_next_state function"""

    def setUp(self):
        self.p = sim.Particle(4, 2, 3)
        self.g = pytpc.gases.Gas(4, 2, 41.8, 150.)
        self.ef = numpy.array([0, 0, 15e3])
        self.bf = numpy.array([0, 0, -1])
        self.tstep = 5e-11

    @property
    def args(self):
        return self.p, self.g, self.ef, self.bf, self.tstep

    def test_zero_energy(self):
        """If the energy is 0, the particle should stay still"""
        self.p.energy = 0
        old_sv = self.p.state_vector
        new_sv = sim.find_next_state(*self.args)
        self.assertTrue(numpy.array_equal(old_sv, new_sv),
                        msg='state vector changed')


class TestThreshold(unittest.TestCase):
    """Tests for sim.threshold function"""

    def test_above(self):
        self.assertEqual(sim.threshold(40., 20.), 40.)

    def test_below(self):
        self.assertEqual(sim.threshold(10., 20.), 20.)

    def test_equal(self):
        self.assertEqual(sim.threshold(10., 10.), 10.)


class TestDriftVelocityVector(unittest.TestCase):
    """Tests for drift_velocity_vector"""

    def setUp(self):
        self.vd = 5
        self.bfield = 10
        self.efield = 2
        self.tilt = 10*degrees
        self.ot = self.bfield / self.efield * self.vd

    def test_zero_angle(self):
        vdv = sim.drift_velocity_vector(self.vd, self.efield, self.bfield, 0)
        self.assertEqual(vdv[0], 0.)
        self.assertEqual(vdv[1], 0.)

    def test_no_bfield(self):
        vdv = sim.drift_velocity_vector(self.vd, self.efield, 0, self.tilt)
        self.assertEqual(vdv[0], 0.)
        self.assertEqual(vdv[1], 0.)

    def test_perpendicular(self):
        vdv = sim.drift_velocity_vector(self.vd, self.efield, self.bfield, pi/2)
        xcomp = -self.vd / (1 + self.ot**2) * self.ot
        ycomp = 0.0
        zcomp = self.vd / (1 + self.ot**2)
        self.assertAlmostEqual(vdv[0], xcomp)
        self.assertAlmostEqual(vdv[1], ycomp)
        self.assertAlmostEqual(vdv[2], zcomp)

    def test_magnitude(self):
        vdv = sim.drift_velocity_vector(self.vd, self.efield, self.bfield, self.tilt)
        self.assertAlmostEqual(numpy.linalg.norm(vdv), self.vd, delta=0.1)


if __name__ == '__main__':
    unittest.main()