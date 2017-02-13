"""Unit tests for simulation.py"""

from __future__ import division, print_function
import unittest
import pytpc.simulation as sim
import pytpc.gases
import numpy
import numpy.testing as nptest
import pandas as pd
from pytpc.constants import *


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
        self.ot = self.bfield / self.efield * self.vd * 1e4

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

    def test_unit_vector_magnitude(self):
        vdv = sim.drift_velocity_vector(1.0, self.efield, self.bfield, self.tilt)
        self.assertAlmostEqual(numpy.linalg.norm(vdv), 1.0, delta=1e-1)

    def test_shape(self):
        vdv = sim.drift_velocity_vector(self.vd, self.efield, self.bfield, self.tilt)
        self.assertEqual(vdv.shape, (3,))


if __name__ == '__main__':
    unittest.main()