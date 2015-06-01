import unittest
import numpy as np
from numpy import pi
import numpy.testing as nptest

import pytpc.utilities as util


class TestEulerMatrix(unittest.TestCase):
    def test_determinant(self):
        for phi in np.linspace(0, 2*pi, 6):
            for theta in np.linspace(0, 2*pi, 6):
                for psi in np.linspace(0, 2*pi, 6):
                    eumat = util.euler_matrix(phi, theta, psi)
                    self.assertAlmostEqual(np.linalg.det(eumat), 1.0,
                                           msg='(phi, theta, psi) = ({}, {}, {})'.format(phi, theta, psi))

    def test_orthogonality(self):
        for phi in np.linspace(0, 2*pi, 6):
            for theta in np.linspace(0, 2*pi, 6):
                for psi in np.linspace(0, 2*pi, 6):
                    eumat = util.euler_matrix(phi, theta, psi)
                    nptest.assert_allclose(eumat.dot(eumat.T), np.eye(3), atol=1e-5, rtol=1e-5)
                    nptest.assert_allclose(eumat.T.dot(eumat), np.eye(3), atol=1e-5, rtol=1e-5)


class TestConstrainAngle(unittest.TestCase):
    def test_constraint(self):
        for ang in np.linspace(-8*pi, 8*pi, 100):
            c_ang = util.constrain_angle(ang)
            self.assertAlmostEqual(abs(np.sin(ang)), abs(np.sin(c_ang)))
            self.assertAlmostEqual(abs(np.cos(ang)), abs(np.cos(c_ang)))
            self.assertLessEqual(c_ang, 2*pi)
            self.assertGreaterEqual(c_ang, 0)

if __name__ == '__main__':
    unittest.main()
