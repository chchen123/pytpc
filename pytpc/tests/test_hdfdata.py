import unittest
import numpy as np
import numpy.testing as nptest

import pytpc.hdfdata as hdfdata
from pytpc.evtdata import Event

from itertools import cycle


class TestDataPacking(unittest.TestCase):

    def setUp(self):
        self.evt_id = 42
        self.timestamp = 1234567890
        junk_evt = Event()
        dt = junk_evt.dt
        self.traces = np.zeros(64, dtype=dt)

        self.traces['cobo'] = 0
        self.traces['asad'] = 1
        self.traces['aget'] = 2
        self.traces['channel'] = np.arange(64)
        self.traces['pad'] = np.arange(64)
        self.traces['data'] = np.fromiter(cycle(range(4906)), dtype='uint16', count=64*512).reshape(64, 512)

    def test_packing(self):
        evt = Event(self.evt_id, self.timestamp)
        evt.traces = self.traces

        packed = hdfdata.HDFDataFile._pack_get_event(evt)

        nptest.assert_equal(packed[:, 0], evt.traces['cobo'])
        nptest.assert_equal(packed[:, 1], evt.traces['asad'])
        nptest.assert_equal(packed[:, 2], evt.traces['aget'])
        nptest.assert_equal(packed[:, 3], evt.traces['channel'])
        nptest.assert_equal(packed[:, 4], evt.traces['pad'])
        nptest.assert_equal(packed[:, 5:], evt.traces['data'])

    def test_unpacking(self):
        packed = np.zeros((64, 512+5))
        packed[:, 0] = self.traces['cobo']
        packed[:, 1] = self.traces['asad']
        packed[:, 2] = self.traces['aget']
        packed[:, 3] = self.traces['channel']
        packed[:, 4] = self.traces['pad']
        packed[:, 5:] = self.traces['data']

        evt = hdfdata.HDFDataFile._unpack_get_event(self.evt_id, self.timestamp, packed)

        nptest.assert_equal(evt.traces, self.traces)
        self.assertEqual(evt.evt_id, self.evt_id)
        self.assertEqual(evt.timestamp, self.timestamp)

    def test_roundtrip(self):
        evt = Event(self.evt_id, self.timestamp)
        evt.traces = self.traces
        packed = hdfdata.HDFDataFile._pack_get_event(evt)
        new_evt = hdfdata.HDFDataFile._unpack_get_event(evt.evt_id, evt.timestamp, packed)

        self.assertEqual(evt.evt_id, new_evt.evt_id)
        self.assertEqual(evt.timestamp, new_evt.timestamp)
        nptest.assert_equal(evt.traces, new_evt.traces)
