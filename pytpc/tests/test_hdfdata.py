import unittest
import numpy as np
import numpy.testing as nptest
import tempfile
import os
import h5py

import pytpc.hdfdata as hdfdata
from pytpc.evtdata import Event

from itertools import cycle


def make_test_data():
    evt_id = 42
    timestamp = 1234567890
    junk_evt = Event()
    dt = junk_evt.dt
    traces = np.zeros(64, dtype=dt)

    traces['cobo'] = 0
    traces['asad'] = 1
    traces['aget'] = 2
    traces['channel'] = np.arange(64)
    traces['pad'] = np.arange(64)
    traces['data'] = np.fromiter(cycle(range(4906)), dtype='uint16', count=64*512).reshape(64, 512)

    return evt_id, timestamp, traces


class TestDataPacking(unittest.TestCase):

    def setUp(self):
        self.evt_id, self.timestamp, self.traces = make_test_data()

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


class TestIO(unittest.TestCase):

    def setUp(self):
        evt_id, timestamp, traces = make_test_data()
        self.evt = Event(evt_id, timestamp)
        self.evt.traces = traces

    def test_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.h5')
            with hdfdata.HDFDataFile(filename, 'w') as hfile:
                group_name = hfile.get_group_name
                hfile.write_get_event(self.evt)

            with h5py.File(filename, 'r') as h5file:
                self.assertIn(group_name, h5file)
                gp = h5file[group_name]
                self.assertIn(str(self.evt.evt_id), gp)
                ds = gp[str(self.evt.evt_id)]

                self.assertTupleEqual(ds.shape, (self.evt.traces.shape[0], 512+5))

                self.assertEqual(ds.compression, 'gzip')
                self.assertTrue(ds.shuffle)

                self.assertEqual(ds.attrs['evt_id'], self.evt.evt_id)
                self.assertEqual(ds.attrs['timestamp'], self.evt.timestamp)

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.h5')
            with hdfdata.HDFDataFile(filename, 'w') as hfile:
                hfile.write_get_event(self.evt)

            with hdfdata.HDFDataFile(filename, 'r') as hfile:
                read_evt = hfile.read_get_event(self.evt.evt_id)

            self.assertEqual(read_evt.evt_id, self.evt.evt_id)
            self.assertEqual(read_evt.timestamp, self.evt.timestamp)
            nptest.assert_equal(read_evt.traces, self.evt.traces)


class TestContextManager(unittest.TestCase):

    def setUp(self):
        evt_id, timestamp, traces = make_test_data()
        self.evt = Event(evt_id, timestamp)
        self.evt.traces = traces

    def test_context_manager_closes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test.h5')
            with hdfdata.HDFDataFile(filename, 'w') as hfile:
                hfile.write_get_event(self.evt)

            self.assertRaises(Exception, hfile.read_get_event, self.evt.evt_id)
