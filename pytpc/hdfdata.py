import h5py
import numpy as np
from .evtdata import Event


class HDFDataFile(object):
    """Interface class to AT-TPC data stored in HDF5 files.

    This can be used to read full GET events from an HDF5 file. This should *not* be used to read the peaks-only files,
    as those are in a different format.

    **WARNING:** If you open a file in `'w'` mode, all data in that file will be deleted! Open in `'a'` mode to append.

    It is **very** important to close the file safely if you have it open for writing. If you do not close the file
    before your code ends, it might become corrupted. The safest way to use this class is with a context manager, like
    this::

        with pytpc.HDFDataFile('/path/to/file.h5', 'r') as hfile:
            # work with the file
            evt = h5file[0]  # for example

        # The rest of your code, which doesn't need the file, follows after the 'with' statement

    Once you leave the scope of the `with` statement, the file will be automatically closed safely.

    The data is stored in the HDF5 files as one table per event. The tables are stored in the group identified by
    `get_group_name`, and each dataset is named using the event ID. For example, the data for event 4 is stored in a
    table at `/get/4` by default since the default `get_group_name` is `'/get'`.

    In the data tables, rows correspond to channels and columns contain metadata and the digitized traces. This means
    that the shape of each table is N x 517, where N is the number of channels hit in that event and 517 equals 512
    time buckets plus 5 more values to identify the CoBo, AsAd, AGET, channel, and pad number. Each row basically
    looks like this:

        +------+------+------+---------+-----+------+------+------+-----+--------+
        | CoBo | AsAd | AGET | Channel | Pad | TB 1 | TB 2 | TB 3 | ... | TB 511 |
        +------+------+------+---------+-----+------+------+------+-----+--------+

    Then again, if you use this class, you won't really need to worry about those details since it's all taken care of
    already!

    Parameters
    ----------
    fp : string
        The path to the HDF5 file.
    open_mode : string, optional
        An open mode for the file, e.g. 'r' for read-only, 'a' for read/append, or 'w' for truncate and write. Use any
        valid open mode for an h5py file object.
    get_group_name : string, optional
        The name of the group in the HDF5 file containing GET events. Usually just keep the default value of '/get'.
    canonical_evtid_path : str, optional
        Path to an HDF5 file containing a table mapping a canonical event ID to the event ID for each CoBo. This
        can be used to correct for CoBos that dropped triggers.
    canonical_evtid_key : str, optional
        The key to read in the canonical event ID file. The default value is 'canonical_evtids'.
    """

    def __init__(self, fp, open_mode='r', get_group_name='/get', canonical_evtid_path=None,
                 canonical_evtid_key='canonical_evtids'):
        self.fp = h5py.File(fp, mode=open_mode)
        self.get_group_name = get_group_name

        if canonical_evtid_path is not None:
            with h5py.File(canonical_evtid_path, 'r') as hf:
                self.canonical_evtid_table = hf[canonical_evtid_key][:]
        else:
            self.canonical_evtid_table = None

        self.fp.require_group(self.get_group_name)

    def close(self):
        """Close the file if it is open."""
        try:
            self.fp.close()
        except ValueError:
            pass

    @staticmethod
    def _unpack_get_event(evtid, timestamp, data):
        """Unpack the data table from the HDF file, and create an Event object.

        Parameters
        ----------
        evtid : int
            The event ID. This will be set in the resulting Event object.
        timestamp : int
            The timestamp value, which will also be set in the resulting Event object.
        data : array-like (or h5py Dataset)
            The data. The shape should be (N, 517) where N is the number of nonzero traces and 517 represents the
            columns (cobo, asad, aget, channel, pad, tb1, tb2, tb3, ..., tb511) from the data.

        Returns
        -------
        evt : pytpc.Event
            The Event object, as read from the file.
        """
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
        """Pack the provided event into a table for storage in the HDF5 file.

        Parameters
        ----------
        evt : pytpc.Event
            The event object to pack.

        Returns
        -------
        np.array
            The data from the event. The shape is (N, 517) where N is the number of nonzero traces and 517 represents
            the columns (cobo, asad, aget, channel, pad, tb1, tb2, tb3, ..., tb511).
        """
        t = evt.traces
        packed = np.zeros((len(t), 517), dtype='int16')
        packed[:, 0] = t['cobo']
        packed[:, 1] = t['asad']
        packed[:, 2] = t['aget']
        packed[:, 3] = t['channel']
        packed[:, 4] = t['pad']
        packed[:, 5:] = t['data']
        return packed

    def read_get_event(self, i):
        """Read the event identified by `i` from the file.

        This is the function to call to get an event from the HDF5 file. It will read the event from the file and return
        a pytpc Event object. The function will look for the event in the HDF5 group listed in `self.get_group_name`,
        so that must be set correctly on initializing the `HDFDataFile` object.

        Parameters
        ----------
        i : int (or str-like see below)
            The identifier for the event in the file. The code will look for the event at `/get/i` in the file if `/get`
            is the `get_group_name` set on the `HDFDataFile` object. In principle, this identifier `i` should be an
            integer representing the event ID, but it could actually be anything that can be cast to a str.

        Returns
        -------
        pytpc.Event
            The event, as rebuilt from the data stored in the file.

        Raises
        ------
        KeyError
            If there is no dataset called `i` in the HDF5 group `self.get_group_name` (i.e. the event ID provided was
            not valid).
        """
        gp = self.fp[self.get_group_name]

        if self.canonical_evtid_table is not None:
            # Use the canonical event ID table to correct for a dropped trigger
            # in some CoBo. Read fragments of each required CoBo event, and then
            # reconstruct the full event.
            cobo_evt_ids = self.canonical_evtid_table[i]
            cobo_evts = {n: gp[str(n)][:] for n in np.unique(cobo_evt_ids)}

            evt_chunks = []
            for cobo_id, cobo_evt_id in enumerate(cobo_evt_ids):
                cobo_evt = cobo_evts[cobo_evt_id]
                chunk = cobo_evt[np.where(cobo_evt[:, 0] == cobo_id)]
                evt_chunks.append(chunk)

            rawevt = np.concatenate(evt_chunks, axis=0)
            evt_id = i
            timestamp = 0  # FIXME: make real?

        else:
            # There is no canonical event ID table, so just read the event.
            ds = gp[str(i)]
            evt_id = ds.attrs.get('evt_id', 0)
            timestamp = ds.attrs.get('timestamp', 0)
            rawevt = ds[:]

        return self._unpack_get_event(evt_id, timestamp, rawevt)

    def write_get_event(self, evt):
        """Write the given Event object to the file.

        This function should be used to append an event to the HDF5 file. This will naturally only work if the file
        is opened in a writable mode.

        The generated dataset will be placed in the HDF5 file at `/get/evt_id` where `/get` is the `get_group_name`
        property of the `HDFDataFile` object and `evt_id` is the event ID property of the given Event object.

        The data is written with gzip compression and the HDF5 shuffle filter turned on. These filters are described
        in the HDF5 documentation. The format of the dataset is described in this class's documentation.

        Parameters
        ----------
        evt : pytpc.Event
            The event to write to the file.

        Raises
        ------
        RuntimeError
            If a dataset for this event already exists. This could also happen if the event ID is not unique in the run.

        """
        evt_id = evt.evt_id
        ts = evt.timestamp
        gp = self.fp[self.get_group_name]
        flat = self._pack_get_event(evt)
        ds = gp.create_dataset(str(evt_id), data=flat, compression='gzip', shuffle=True)
        ds.attrs['evt_id'] = evt_id
        ds.attrs['timestamp'] = ts
        self.fp.flush()

    def evtids(self):
        """Returns an iterator over the set of event IDs in the file, as integers.
        """
        return (int(key) for key in self.fp[self.get_group_name])

    def __len__(self):
        return len(self.fp[self.get_group_name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return (self.read_get_event(n) for n in self.fp[self.get_group_name])

    def __getitem__(self, i):
        return self.read_get_event(i)
