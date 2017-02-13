from unittest import TestCase
from pytpc.constants import degrees, pi
import numpy as np
import numpy.testing as nptest
from ..paramgen import make_random_beam


def normalize(vec):
    return vec / np.linalg.norm(vec)


class MakeRandomBeamTestCase(TestCase):
    def test_results(self):
        max_angle = 2 * degrees
        focus_loc = 1.28
        vertex_loc = 0.4
        window_loc = 1.0

        np.random.seed(0)
        beam, vertex, window, transform = make_random_beam(max_angle, focus_loc, vertex_loc, window_loc)

        beam_focus = np.array([0., 0., focus_loc])
        expect_beam_vec = normalize(vertex - beam_focus)

        nptest.assert_allclose(beam, expect_beam_vec)

        self.assertAlmostEqual(vertex[2], vertex_loc)
        self.assertAlmostEqual(window[2], window_loc)
