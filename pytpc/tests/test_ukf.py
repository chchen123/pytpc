import unittest
import pytpc.ukf
import numpy.testing as npt
import numpy as np


def predict_func(x):
    pass


def measure_func(z):
    pass

def ts_func(z):
    pass


class TestUnscentedKalmanFilter(unittest.TestCase):
    def setUp(self):
        self.dim_x = 4
        self.dim_z = 2
        self.f = pytpc.ukf.UnscentedKalmanFilter(self.dim_x, self.dim_z,
                                                 predict_func, measure_func,
                                                 ts_func)

    def test_constructor(self):
        f = self.f
        self.assertEqual(f._dim_x, self.dim_x)
        self.assertEqual(f._dim_z, self.dim_z)

        self.assertIs(f.fx, predict_func)
        self.assertIs(f.hx, measure_func)

        self.assertTupleEqual(f.Q.shape, (self.dim_x, self.dim_x))
        self.assertTupleEqual(f.R.shape, (self.dim_z, self.dim_z))
        self.assertTupleEqual(f.x.shape, (self.dim_x,))
        self.assertTupleEqual(f.P.shape, (self.dim_x, self.dim_x))

    def test_find_weights(self):
        for n in range(1, 10):
            for k in range(10):
                w = self.f.find_weights(n, k)

                w0_exp = k/(n+k)
                self.assertEqual(w[0], w0_exp)

                wi_exp = np.full(np.size(w)-1, 0.5 / (n+k))
                npt.assert_allclose(w[1:], wi_exp)


if __name__ == '__main__':
    unittest.main()
