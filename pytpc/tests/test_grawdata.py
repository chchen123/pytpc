import unittest
import numpy as np
import numpy.testing as nptest

import pytpc.grawdata

class TestDataUnpacking(unittest.TestCase):

    def test_unpack_partial_readout(self):
        maxval = (3 << 30) | (67 << 23) | (511 << 14) | 4095

        aget, ch, tb, samp = pytpc.grawdata.GRAWFile._unpack_data_partial_readout(maxval)
        self.assertEqual(aget, 3, msg='invalid aget')
        self.assertEqual(ch, 67, msg='invalid channel')  # This is intentional since this is the width of 7 bits
        self.assertEqual(tb, 511, msg='invalid tb')
        self.assertEqual(samp, 4095, msg='invalid sample')

    
