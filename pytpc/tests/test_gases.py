import unittest
import pytpc.gases
import pytpc.relativity as rel
from pytpc.constants import *


class TestGenericGas(unittest.TestCase):
    def setUp(self):
        self.gas = pytpc.gases.GenericGas(molar_mass=10., num_electrons=4.,
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


class TestGas(unittest.TestCase):
    def setUp(self):
        self.gas = pytpc.gases.Gas(molar_mass=10., pressure=200.)

    def test_density(self):
        res = self.gas.density
        expect = self.gas.pressure / 760. * self.gas.molar_mass / 24040.
        self.assertEqual(res, expect)


class TestInterpolatedGas(unittest.TestCase):

    def setUp(self):
        self.gas = pytpc.gases.InterpolatedGas('helium', 200)

    def test_energy_loss(self):
        for en in range(0, 10):
            res = self.gas.energy_loss(en, 4, 2)
            exp = self.gas.spline(en) * self.gas.density * 100
            self.assertAlmostEqual(res, exp)


class TestInterpolatedGasMixture(unittest.TestCase):

    def setUp(self):
        self.heco = pytpc.gases.InterpolatedGasMixture(200, ('helium', 0.5),
                                                            ('carbon_dioxide', 0.5))
        self.he = pytpc.gases.InterpolatedGas('helium', 100)
        self.co = pytpc.gases.InterpolatedGas('carbon_dioxide', 100)

    def test_density(self):
        res = self.heco.density
        exp = self.he.density + self.co.density
        self.assertAlmostEqual(res, exp)

    def test_molar_mass(self):
        res = self.heco.molar_mass
        exp = self.he.molar_mass * 0.5 + self.co.molar_mass * 0.5
        self.assertAlmostEqual(res, exp)

    def test_pressure(self):
        total = self.heco.pressure
        comps = sum([g.pressure for g, p in self.heco.components])
        self.assertAlmostEqual(comps, total)


class TestBethe(unittest.TestCase):
    """Tests for pytpc.gases.bethe function"""

    def setUp(self):
        self.gas = pytpc.gases.GenericGas(4, 2, 41.8, 100)

    def test_zero_beta(self):
        self.assertEqual(pytpc.gases.bethe(0., 2, self.gas.electron_density_per_m3, self.gas.mean_exc_pot),
                         float('inf'))

    def test_large_beta(self):
        self.assertAlmostEqual(pytpc.gases.bethe(1-1e-10, 2, self.gas.electron_density_per_m3, self.gas.mean_exc_pot),
                               0., delta=0.1)

    def test_high_pressure(self):
        self.gas.pressure = 1e100
        self.assertGreater(pytpc.gases.bethe(0.2, 2, self.gas.electron_density_per_m3, self.gas.mean_exc_pot), 1e10)

    def test_zero_pressure(self):
        self.gas.pressure = 0
        self.assertEqual(pytpc.gases.bethe(0.2, 2, self.gas.electron_density_per_m3, self.gas.mean_exc_pot), 0.0)

    def test_low_pressure(self):
        self.gas.pressure = 1e-3
        self.assertLess(pytpc.gases.bethe(0.2, 2, self.gas.electron_density_per_m3, self.gas.mean_exc_pot), 1e-3)


class TestHeliumGas(unittest.TestCase):
    """Tests the pytpc.gases.HeliumGas class"""
    def setUp(self):
        self.he = pytpc.gases.HeliumGas(100.)

    def test_eloss_with_4he(self):
        en = 1.
        m = 4
        z = 2
        res = self.he.energy_loss(en, m, z)
        bth = pytpc.gases.bethe(rel.beta(en, m*p_mc2), z, self.he.electron_density_per_m3, self.he.mean_exc_pot)
        self.assertAlmostEqual(res, bth, delta=10)

    def test_eloss_with_proton(self):
        en = 1.
        m = 1
        z = 1
        res = self.he.energy_loss(en, m, z)
        bth = pytpc.gases.bethe(rel.beta(en, m*p_mc2), z, self.he.electron_density_per_m3, self.he.mean_exc_pot)
        self.assertAlmostEqual(res, bth, delta=10)

    def test_eloss_bad_proj(self):
        en = 1.
        m = 100
        z = 100
        self.assertRaises(ValueError, self.he.energy_loss, en, m, z)

    def test_eloss_low_en(self):
        en = 1e-8
        m = 4
        z = 2
        res = self.he.energy_loss(en, m, z)
        self.assertGreater(res, 1e2)

    def test_eloss_high_en(self):
        en = 1e3
        m = 4
        z = 2
        res = self.he.energy_loss(en, m, z)
        self.assertLess(res, 0.1)

class TestHeCO2Gas(unittest.TestCase):
    """Tests the pytpc.gases.HeCO2Gas class"""
    def setUp(self):
        self.he = pytpc.gases.HeCO2Gas(100.)

    def test_eloss_with_4he(self):
        en = 1.
        m = 4
        z = 2
        res = self.he.energy_loss(en, m, z)
        bth = pytpc.gases.bethe(rel.beta(en, m*p_mc2), z, self.he.electron_density_per_m3, self.he.mean_exc_pot)
        self.assertAlmostEqual(res, bth, delta=10)

    def test_eloss_bad_proj(self):
        en = 1.
        m = 100
        z = 100
        self.assertRaises(ValueError, self.he.energy_loss, en, m, z)

    def test_eloss_high_en(self):
        en = 1e3
        m = 4
        z = 2
        res = self.he.energy_loss(en, m, z)
        self.assertLess(res, 0.1)

if __name__ == '__main__':
    unittest.main()
