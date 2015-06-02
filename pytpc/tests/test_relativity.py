"""Unit tests for relativity.py"""

from __future__ import division, print_function
import unittest
import pytpc.relativity as rel
import pytpc
from math import sqrt
from pytpc.constants import *
from pytpc.utilities import constrain_angle
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


class TestElasticScatter_IdenticalParticles(unittest.TestCase):
    """Tests for the elastic scattering function"""

    def setUp(self):
        self.proj = pytpc.Particle(4, 2, 2)
        self.target = pytpc.Particle(4, 2, 0)
        self.cm_angle = pi / 4
        self.azi = pi / 3

    def test_polar(self):
        for cm_angle in numpy.linspace(0.01, pi, 20, endpoint=False):
            ejec, recoil = rel.elastic_scatter(self.proj, self.target, cm_angle, self.azi)
            self.assertAlmostEqual(ejec.polar + recoil.polar, pi/2, delta=2.0)

    def test_angles_beam90_reacazi0(self):
        for azi in numpy.linspace(0, 2*pi, 100):
            proj = pytpc.Particle(4, 2, 2, polar=pi/2, azimuth=azi)
            ejec, recoil = rel.elastic_scatter(proj, self.target, pi/2, 0)
            self.assertAlmostEqual(ejec.azimuth, azi, places=3)
            self.assertAlmostEqual(recoil.azimuth, azi, places=3)
            self.assertAlmostEqual(abs(ejec.polar - recoil.polar), pi/2, places=2)

    def test_angles_beam90_reacazi90(self):
        for azi in numpy.linspace(0, 2*pi, 100):
            proj = pytpc.Particle(4, 2, 2, polar=pi/2, azimuth=azi)
            ejec, recoil = rel.elastic_scatter(proj, self.target, pi/2, pi/2)
            self.assertAlmostEqual(ejec.polar, pi/2)
            self.assertAlmostEqual(recoil.polar, pi/2)
            self.assertAlmostEqual(abs(numpy.sin(ejec.azimuth - azi)), numpy.sin(pi/4), places=2)
            self.assertAlmostEqual(abs(numpy.sin(recoil.azimuth - azi)), numpy.sin(pi/4), places=2)
            # self.assertAlmostEqual(abs(ejec.polar - recoil.polar), pi/2, places=2)

    def test_azimuth(self):
        for azi in numpy.linspace(0, 2*pi, 20):
            ejec, recoil = rel.elastic_scatter(self.proj, self.target, self.cm_angle, azi)
            self.assertAlmostEqual(ejec.azimuth, azi)
            self.assertAlmostEqual(recoil.azimuth, constrain_angle(azi + pi))

    def test_energy_small_angle(self):
        ejec, recoil = rel.elastic_scatter(self.proj, self.target, 0.01, self.azi)
        self.assertAlmostEqual(recoil.energy, 0, places=3)

    def test_energy_equal(self):
        ejec, recoil = rel.elastic_scatter(self.proj, self.target, pi/2, self.azi)
        self.assertAlmostEqual(recoil.energy, ejec.energy)

    def test_energy_large_angle(self):
        ejec, recoil = rel.elastic_scatter(self.proj, self.target, pi-0.01, self.azi)
        self.assertAlmostEqual(ejec.energy, 0, places=3)


class TestElasticScatter_HeavyTarget(unittest.TestCase):
    """Tests for the elastic scattering function"""

    def setUp(self):
        self.proj = pytpc.Particle(4, 2, 2)       # helium-4
        self.target = pytpc.Particle(1e8, 100, 0)  # nonphysical, but hopefully fixed
        self.cm_angle = pi / 4
        self.azi = pi / 3

    def test_energy(self):
        ejec, recoil = rel.elastic_scatter(self.proj, self.target, self.cm_angle, self.azi)
        self.assertAlmostEqual(recoil.energy, 0, places=4)
        self.assertAlmostEqual(ejec.energy, self.proj.energy, places=4)


if __name__ == '__main__':
    unittest.main()
