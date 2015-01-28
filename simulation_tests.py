import unittest
import simulation as sim
from math import sqrt, sin, cos, atan2
import numpy
from constants import *


class TestGas(unittest.TestCase):
    def setUp(self):
        self.gas = sim.Gas(molar_mass=10., num_electrons=4.,
                                mean_exc_pot=8., pressure=200.)

    def test_density(self):
        res = self.gas.density
        expect = self.gas.pressure / 760. * self.gas.molar_mass / 24040.
        self.assertEqual(res, expect)

    def test_electron_density(self):
        res = self.gas.electron_density
        expect = N_avo * self.gas.num_electrons * self.gas.density / self.gas.molar_mass
        self.assertEqual(res, expect)

    def test_electron_density_per_m3(self):
        res = self.gas.electron_density_per_m3
        expect = N_avo * self.gas.num_electrons * self.gas.density / self.gas.molar_mass * 1e6
        self.assertEqual(res, expect)


class TestParticle(unittest.TestCase):
    def setUp(self):
        self.en_i = 3
        self.Z_i = 2
        self.A_i = 4
        self.pos_i = (1., 1., 1.)
        self.azi_i = pi/2
        self.pol_i = pi/3
        self.p = sim.Particle(self.A_i, self.Z_i, self.en_i, self.pos_i, self.azi_i,
                                   self.pol_i)

    def test_mass(self):
        exp_mass = self.p.mass_num * p_mc2
        self.assertEqual(self.p._mass, exp_mass)
        self.assertEqual(self.p.mass, exp_mass)
        self.assertEqual(self.p.mass_kg, exp_mass * MeVtokg)

    def test_energy(self):
        exp_energy = self.en_i * self.A_i
        self.assertEqual(self.p._energy, exp_energy)
        self.assertEqual(self.p.energy, exp_energy)
        self.assertEqual(self.p.energy_per_particle, exp_energy / self.A_i)
        self.assertEqual(self.p.energy_j, exp_energy * 1e6 * e_chg)

    def test_energy_setter(self):
        new_en = (self.en_i + 1) * self.A_i
        self.p.energy = new_en
        self.assertEqual(self.p._energy, new_en)

    def test_energy_per_particle_setter(self):
        new_en_per_u = (self.en_i / self.A_i) + 1
        self.p.energy_per_particle = new_en_per_u
        self.assertEqual(self.p._energy / self.p.mass_num, new_en_per_u)

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
        self.assertEqual(self.p.velocity.all(), exp_vel.all())

    def test_velocity_setter(self):
        new_vel = 2 * self.p.velocity
        vx, vy, vz = new_vel
        self.p.velocity *= 2
        self.assertEqual(self.p.velocity.all(), new_vel.all())

        exp_azi = atan2(vy, vx)
        exp_pol = atan2(sqrt(vx**2 + vy**2), vz)
        self.assertEqual(self.p.azimuth, exp_azi)
        self.assertEqual(self.p.polar, exp_pol)

        exp_en = (1/sqrt(1 - numpy.linalg.norm(new_vel)**2 / c_lgt**2) - 1)*self.p.mass
        self.assertEqual(self.p.energy, exp_en)


if __name__ == '__main__':
    unittest.main()
