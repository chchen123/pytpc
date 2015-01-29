import unittest
import relativity as rel
from math import sqrt
from constants import *
import numpy


class TestGamma(unittest.TestCase):
    """Tests for relativity.gamma(v) function
    """

    def test_v_gt_c(self):
        # test with v > c
        v_big = numpy.array([3e10, 3e10, 3e10])
        self.assertRaises(ValueError, rel.gamma, v_big)

    def test_v_eq_c(self):
        self.assertRaises(ValueError, rel.gamma, c_lgt)

    def test_v_eq_zero(self):
        v_zero = numpy.zeros(3)
        self.assertEqual(rel.gamma(v_zero), 1.0)

    def test_with_scalar(self):
        v = 3.0e5
        exp = 1 / sqrt(1 + v**2 / c_lgt**2)
        self.assertAlmostEqual(rel.gamma(v), exp, places=4)


class TestBeta(unittest.TestCase):
    """Tests for relativity.beta function"""

    def test_zero_mass(self):
        self.assertEqual(rel.beta(4, 0), 1)

    def test_zero_energy(self):
        self.assertEqual(rel.beta(0.0, 4), 0)

    def test_zero_mass_zero_energy(self):
        self.assertRaises(ValueError, rel.beta, 0, 0)

    def test_negative_mass(self):
        self.assertRaises(ValueError, rel.beta, en=4.0, mass=-4.0)

    def test_negative_energy(self):
        self.assertRaises(ValueError, rel.beta, en=-4.0, mass=-4.0)

    def test_negative_mass_and_energy(self):
        self.assertRaises(ValueError, rel.beta, en=-4.0, mass=-4.0)

    def test_value(self):
        for e in range(1, 40):
            for m in range(1, 40):
                exp = (sqrt(e) * sqrt(e + 2*m)) / (e + m)
                self.assertAlmostEqual(rel.beta(en=e, mass=m), exp, places=5)

if __name__ == '__main__':
    unittest.main()
