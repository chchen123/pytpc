import h5py
import numpy as np
from .evtdata import Event


class HDFDataFile(object):

    def __init__(self, fp, open_mode='r', get_group_name='/get'):
        self.fp = h5py.File(fp, mode=open_mode)
        self.get_group_name = get_group_name

        self.fp.require_group(self.get_group_name)

    def close(self):
        try:
            self.fp.close()
        except ValueError:
            pass

    @staticmethod
    def _unpack_get_event(evtid, timestamp, data):
        evt = Event(evtid, timestamp)

        evt.traces = np.zeros(data.shape[0], dtype=evt.dt)
        evt.traces['cobo'] = data[:, 0]
        evt.traces['asad'] = data[:, 1]
        evt.traces['aget'] = data[:, 2]
        evt.traces['channel'] = data[:, 3]
        evt.traces['pad'] = data[:, 4]
        evt.traces['data'] = data[:, 5:]

        return evt

    @staticmethod
    def _pack_get_event(evt):
        t = evt.traces
        return np.column_stack((t['cobo'], t['asad'], t['aget'], t['channel'], t['pad'], t['data']))

    def read_get_event(self, i):
        gp = self.fp[self.get_group_name]
        ds = gp[str(i)]
        evt_id = ds.attrs.get('evt_id', 0)
        timestamp = ds.attrs.get('timestamp', 0)
        return self._unpack_get_event(evt_id, timestamp, ds)

    def write_get_event(self, evt):
        evt_id = evt.evt_id
        ts = evt.timestamp
        gp = self.fp[self.get_group_name]
        flat = self._pack_get_event(evt)
        ds = gp.create_dataset(str(evt_id), data=flat, compression='gzip', shuffle=True)
        ds.attrs['evt_id'] = evt_id
        ds.attrs['timestamp'] = ts
        self.fp.flush()

    def __len__(self):
        return len(self.fp[self.get_group_name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return (self.read_get_event(n) for n in self.fp[self.get_group_name].keys())

    def __getitem__(self, i):
        return self.read_get_event(i)
