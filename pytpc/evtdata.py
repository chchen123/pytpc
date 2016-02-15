"""
evtdata
=======

This module provides functionality for interacting with GET event files. These files contain merged events.

..  Note::
    This module does *not* allow interaction with un-merged GRAW files.

Examples
--------

..  code-block:: python

    from pytpc.evtdata import EventFile, Event

    # Open a file
    ef = EventFile('/path/to/data.evt')

    # Read the fourth event
    evt = ef[4]

    # Iterate over all events
    for evt in ef:
        do_something(evt)

    # Slices are also supported
    for evt in ef[4:20]:
        do_something(evt)

"""

from __future__ import division, print_function
import struct
import numpy as np
import os.path
import warnings
from pytpc.tpcplot import generate_pad_plane
import pytpc.datafile
from scipy.stats import threshold
from scipy.ndimage.filters import gaussian_filter
from pytpc.padplane import find_pad_coords, padcenter_dict
from math import sin, cos


class EventFile (pytpc.datafile.DataFile):
    """Represents a merged GET event file.

    This class provides an interface to merged event files from the GET electronics. It is capable of opening
    merged data files and reading events from them.

    When an event file is first opened, its contents are automatically indexed. The generated lookup table is saved
    in the same directory as the event file, but with the extension ".lookup".

    The events can be accessed via subscripting and iteration.

    Parameters
    ----------
    filename : string, optional
        If provided, opens the file located at this path. Otherwise, a file can be opened using the `open` method.

    Attributes
    ----------
    magic : int
        The magic number of the file, used to make sure we've opened a file of the correct type. Its value
        is `0x6e7ef11e`.
    lookup : list
        A lookup table indexing the offset of each event in the file. This is generated automatically.
    current_event : int
        The current event number, or which event the file pointer is at.
    is_open : bool
        Whether a file is currently open
    fp : file
        The file object itself
    """

    def __init__(self, filename=None, open_mode='r'):

        self.evtids = None
        self.magic = 0x6e7ef11e  # The file's magic number

        super().__init__(filename, open_mode)

        if 'r' in open_mode:
            self.fp.seek(0)
            read_magic = struct.unpack('<I', self.fp.read(4))[0]
            if read_magic != self.magic:
                raise IOError("Bad file. Magic number is wrong.")
        elif 'w' in open_mode:
            self.fp.write(self.magic.to_bytes(4, byteorder='little'))

        return

    @staticmethod
    def pack_sample(tb, val):
        """Pack a sample value into three bytes.

        See the documentation of the Event File format for details.

        Parameters
        ----------
        tb : int
            The time bucket
        val : int
            The sample

        Returns
        -------
        joined : int
            The packed sample
        """

        # val is 12-bits and tb is 9 bits. Fit this in 24 bits.
        # Use one bit for parity

        if val < 0:
            parity = 1 << 12
            narrowed = -val
        elif val >= 4095:
            parity = 0
            narrowed = 4095
        else:
            parity = 0
            narrowed = val

        joined = (tb << 15) | narrowed | parity
        return joined

    @staticmethod
    def pack_samples(tb, val):

        parities = np.where(val < 0, 1, 0)
        narrowed = np.clip(abs(val), 0, 4095)
        joined = (tb << 15) | (parities << 12) | narrowed
        return joined

    @staticmethod
    def unpack_sample(packed):
        """Unpacks a packed time bucket / sample pair.

        Parameters
        ----------
        packed : int
            The packed value

        Returns
        -------
        tb : int
            The time bucket
        sample : int
            The sample value
        """

        tb = (packed & 0xFF8000) >> 15
        sample = (packed & 0xFFF)
        parity = (packed & 0x1000) >> 12

        if parity == 1:
            sample *= -1

        return tb, sample

    @staticmethod
    def unpack_samples(packed):
        """ Unpacks an array of time bucket / sample pairs.

        This is really just a modified version of unpack_sample that works with arrays.

        Parameters
        ----------
        packed : array-like
            The packed values

        Returns
        -------
        samples : ndarray
            The sample values, indexed by time bucket
        """

        tbs = (packed & 0xFF8000) >> 15
        samples = (packed & 0xFFF).astype('int16')
        parities = (packed & 0x1000) >> 12

        # Change the parities from (0, 1, ...) to (1, -1, ...) and multiply
        samples *= (-2 * parities + 1)

        result = np.zeros(512)

        result[tbs] = samples

        return result

    def _read(self):
        """Reads an event at the current file position.

        The position is checked by checking the event magic number ``0xEE``.

        Returns
        -------
        Event
            The read event
        """

        assert self.is_open

        # Read the event header
        # This has the following structure:
        #     (magic, size, ID, ts, nTraces)
        hdr = struct.unpack('<BIIQH', self.fp.read(19))

        if hdr[0] != 0xEE:
            # This is not the beginning of the event. Seek back and fail.
            self.fp.seek(-19, 1)
            raise FilePosError(self.fp.name, "Event magic number was wrong.")

        event_size = hdr[1]
        event_id = hdr[2]
        event_ts = hdr[3]
        num_traces = hdr[4]

        new_evt = Event(evt_id=event_id, timestamp=event_ts)
        new_evt.traces = np.zeros((num_traces,), dtype=new_evt.dt)

        # Read the traces

        for n in range(num_traces):
            # Read the trace header. The structure of this is:
            #    (size, cobo, asad, aget, ch, pad)
            th = struct.unpack('<IBBBBH', self.fp.read(10))

            tr = new_evt.traces[n]

            tr['cobo'] = th[1]
            tr['asad'] = th[2]
            tr['aget'] = th[3]
            tr['channel'] = th[4]
            tr['pad'] = th[5]

            # Now read the trace itself
            num_samples = (th[0] - 10) // 3  # (total - header) / size of packed item

            packed = np.hstack((np.fromfile(self.fp, dtype='3u1', count=num_samples),
                                np.zeros((num_samples, 1), dtype='u1'))).view('<u4')

            tr['data'][:] = self.unpack_samples(packed)

        return new_evt

    def write(self, evt):
        evt_magic = 0xee
        size = 19 + len(evt.traces) * 10 + np.count_nonzero(evt.traces['data']) * 3
        evtid = evt.evt_id
        timestamp = evt.timestamp
        num_traces = len(evt.traces)

        evt_hdr = struct.pack('<BIIQH', evt_magic, size, evtid, timestamp, num_traces)
        self.fp.write(evt_hdr)

        for tr in evt.traces:
            tr_size = np.count_nonzero(tr['data']) * 3 + 10
            th = struct.pack('<IBBBBH', tr_size, tr['cobo'], tr['asad'], tr['aget'], tr['channel'], tr['pad'])
            self.fp.write(th)
            nztbs = np.nonzero(tr['data'])[0]
            nzvals = tr['data'][nztbs]
            packed = self.pack_samples(nztbs, nzvals)
            packed.view('<8u1')[:, :3].tofile(self.fp)

    def make_lookup_table(self):
        """ Indexes the open file and generates a lookup table.

        The lookup table is a list of the file offsets of each event, in bytes. This is used by the read_next() and
        read_previous() functions to navigate around the file. The table is returned, but it is also assigned to
        self.lookup.

        Returns
        -------
        lookup : list
            The lookup table
        """

        assert self.is_open

        self.fp.seek(4)

        self.lookup = []
        self.evtids = []

        while True:
            magic = self.fp.read(1)
            if magic == b'\xEE':
                pos = self.fp.tell() - 1
                offset, evtid = struct.unpack('<II', self.fp.read(8))
                self.lookup.append(pos)
                self.evtids.append(evtid)
                self.fp.seek(pos + offset)
            elif magic == b'':
                break
            else:
                print("Bad magic number")
                return None

        self.evtids = np.array(self.evtids)

        self.fp.seek(4)
        self.current_event = 0

        ltfilename = self.fp.name + '.lookup'

        if not os.path.exists(ltfilename):
            ltfile = open(ltfilename, 'w')
            for i, e in zip(self.lookup, self.evtids):
                ltfile.write(str(i) + ',' + str(e) + '\n')
            ltfile.close()

        return self.lookup

    def get_by_event_id(self, evtid):
        """Get the event with the specified event ID from the file.

        This is useful if the events in the file are not in order, or if the event numbering did not begin at 0.

        Parameters
        ----------
        evtid : integer
            The event ID to look for.

        Returns
        -------
        evt : Event
            The event

        Raises
        ------
        KeyError
            If there is no event in the file with the given event ID.
        ValueError
            If the event ID is not unique in the file. (This shouldn't happen, and may indicate corrupt data.)
        """
        idx = np.where(self.evtids == evtid)[0]
        if len(idx) == 1:
            assert len(self.lookup) == len(self.evtids), 'length of lookup tables does not match'
            return self.read_event_by_number(idx[0])
        elif len(idx) == 0:
            raise KeyError('No event with this ID')
        else:
            raise ValueError('More than one event had this ID! (This may indicate corrupt data.)')


class Event:
    """Represents an event from an event file.

    The event is represented as a np array with a complex dtype.

    Parameters
    ----------
    evt_id : int
        The event ID
    timestamp : int
        The event timestamp

    Attributes
    ----------
    evt_id : int
        The event ID
    timestamp : int
        The event timestamp
    dt : np.dtype
        The datatype used for the ndarray storing the traces. This defines fields `cobo`, `asad`, `aget`, `channel`,
        `pad`, and `data`. The `data` is an array of samples indexed by time bucket.
    traces : ndarray
        The event data. The datatype is given above.

    """

    def __init__(self, evt_id=0, timestamp=0):

        assert (evt_id >= 0)
        assert (timestamp >= 0)

        self.dt = np.dtype([('cobo', 'u1'), ('asad', 'u1'), ('aget', 'u1'), ('channel', 'u1'),
                            ('pad', 'u2'), ('data', '512i2')])
        """The data type of the numpy array"""

        self.evt_id = evt_id
        """The event ID"""

        self.timestamp = timestamp
        """The event timestamp"""

        self.traces = np.zeros((0,), dtype=self.dt)
        """The event data. The dtype is :attr:`dt`."""

        return

    def __str__(self):
        s = 'Event {id}, timestamp {ts}.\nContains {tr} traces.'
        return s.format(id=self.evt_id, ts=self.timestamp, tr=len(self.traces))

    def hits(self):
        """Calculate the total activation of each pad.

        The activation is calculated by summing the activation for all time buckets in each channel. This produces
        something similar to the mesh trace that we monitor during experiments.

        Returns
        -------
        hits : ndarray
            An array of hits, indexed by pad number
        """

        hits = self.traces['data'].sum(1)
        pads = self.traces['pad']

        flat_hits = np.zeros(10240)
        for (p, h) in zip(pads, hits):
            flat_hits[p] = h

        return flat_hits

    def xyzs(self, drift_vel=None, clock=None, pads=None, peaks_only=False, return_pads=False):
        """Find the scatter points of the event in space.

        If a drift velocity and write clock frequency are provided, then the result gives the z dimension in
        meters. Otherwise, the z dimension is measured in time buckets.

        Parameters
        ----------
        drift_vel : number or array-like
            The drift velocity in the detector, in cm/us. This can be the scalar magnitude or a vector. A vectorial
            drift velocity can be used to correct for the Lorentz angle.
        clock : int or float, optional
            The write clock rate, in MHz
        pads : ndarray, optional
            An array of pad vertices. If provided, these pads will be used instead of the default pad plane.
        peaks_only : bool, optional
            If True, only the peak of each activation curve will be used.
        return_pads: bool, optional
            If True, also return the pad numbers in the output array. They will be in the last column.

        Returns
        -------
        xyzs : np.ndarray
            A 4D array of points including (x, y, tb or z, activation). If `return_pads` is True, there will also
            be a fifth column containing the pad numbers.

        See also
        --------
        pytpc.simulation.drift_velocity_vector
        """

        if peaks_only:
            traces_copy = np.copy(self.traces)
            for i, tr in enumerate(traces_copy):
                traces_copy[i]['data'][:] = threshold(tr['data'], threshmin=tr['data'].max(), newval=0)
            nz = traces_copy['data'].nonzero()

        else:
            nz = self.traces['data'].nonzero()

        if pads is None:
            pads = generate_pad_plane()

        pcenters = pads.mean(1)

        xys = pcenters[self.traces[nz[0]]['pad']]
        zs = nz[1]
        cs = self.traces['data'][nz]

        if return_pads:
            padnums = self.traces[nz[0]]['pad']
            xyzs = np.column_stack((xys, zs, cs, padnums))
        else:
            xyzs = np.column_stack((xys, zs, cs))

        if drift_vel is not None and clock is not None:
            xyzs = calibrate(xyzs, drift_vel, clock)

        return xyzs


def make_event(pos, de, clock, vd, ioniz, proj_mass, shapetime, offset=0, pad_rot=None):
    """Generate an Event from simulated data.

    This will take the given position and energy loss information and project it onto the pad plane to create
    an event out of it. This can take the drift velocity, shaping time of the electronics, and the Lorentz angle from
    the fields into account. (For the shaping in the electronics, the signal is assumed to be a Gaussian with standard
    deviation equal to the shaping time.

    Parameters
    ----------
    pos : ndarray
        The [x, y, z] positions of the simulated track
    de : ndarray
        The energy loss *per nucleon* for each step. It is important that this have the same length as `pos`.
    clock : number
        The CoBo write clock frequency, in MHz.
    vd : array-like or number
        The drift velocity in the detector. Provide as a vector to account for the Lorentz angle. The units
        should be cm/µs in either case.
    ioniz : number
        The ionization energy of the detector gas, in eV.
    proj_mass : integer
        The projectile mass number
    shapetime : number
        The shaping time, in ns.
    offset : integer, optional
        A time bucket offset. If provided, the signal will be shifted right (toward higher time buckets by this amount.

    Returns
    -------
    evt : Event
        The simulated data as an Event object.

    See Also
    --------
    pytpc.simulation.drift_velocity_vector
    pytpc.simulation.track
    pytpc.simulation.simulate_elastic_scattering_track

    """

    # Find energy deposition for each point
    ne = np.round(de*1e6*proj_mass / ioniz)  # last factor after mass is gain

    # Uncalibrate the position data
    uncal = uncalibrate(pos, vd, clock, offset)
    tbs = uncal[:, -1]

    # Rotate to deal with pad plane rotation
    if pad_rot is not None:
        assert uncal.shape[1] == 3
        rot_mat = np.array([[cos(pad_rot), -sin(pad_rot), 0],
                            [sin(pad_rot),  cos(pad_rot), 0],
                            [0,             0,            1]])
        uncal = np.inner(rot_mat, uncal).T

    # Find the pad for each point
    pca = np.round(find_pad_coords(uncal[:, 0:2]))
    pnums = np.array([padcenter_dict.get(tuple(a), -1) for a in pca])
    unique_pads = set(pnums)
    unique_pads.discard(-1)

    shapetime_tb = shapetime*1e-9 * clock*1e6

    # Build the event
    evt = Event()
    evt.traces = np.zeros(len(unique_pads), dtype=evt.dt)
    for i, p in enumerate(unique_pads):
        idxs = np.where(pnums == p)
        tr = np.zeros(512)
        for t, v in zip(tbs[idxs], ne[idxs]):
            if 0 <= t <= 511:
                tr[t] += v
            else:
                warnings.warn('Event clipped: time bucket overflow/underflow', RuntimeWarning)
        evt.traces[i]['pad'] = p
        evt.traces[i]['data'][:] = gaussian_filter(tr, shapetime_tb, mode='constant')
    return evt


def calibrate(data, drift_vel, clock):
    """Calibrate the data using the drift velocity.

    With a scalar drift velocity, this will simply convert the time buckets into position coordinates. With a vector
    drift velocity, this can also correct the 3D representation for the effects of the magnetic field on the tracks of
    the drift electrons, a.k.a. the Lorentz angle.

    Parameters
    ----------
    data : ndarray, int, float
        The uncalibrated data, in the form [x, y, z, ...]
    drift_vel : number or array-like
        The drift velocity in the gas, in cm/us. If scalar, it will only affect the z values. If it's a vector, all
        spatial points will be transformed accordingly.
    clock : int, float
        The CoBo write clock frequency, in MHz

    Returns
    -------
    ndarray or int or float
        The calibrated data

    See also
    --------
    pytpc.simulation.drift_velocity_vector

    """
    new_data = data.copy()

    if np.isscalar(drift_vel):
        new_data[:, 2] *= drift_vel / clock * 10  # 1 cm/(us.MHz) = 1 cm = 10 mm
    else:
        assert drift_vel.shape == (3,), 'vector drift velocity must have 3 dimensions'
        new_data[:, 2] = 0.0  # This prevents adding the time buckets to the last dimension
        new_data[:, 0:3] += np.outer(data[:, 2], -drift_vel) / clock * 10

    return new_data


def uncalibrate(data, drift_vel, clock, offset=0):
    """Uncalibrate the data using the drift velocity.

    With a scalar drift velocity, this will simply convert the z position coordinates into time buckets. With a vector
    drift velocity, this will project the points down onto the pad plane, taking into account the Lorentz angle from
    the magnetic field.

    Parameters
    ----------
    data : ndarray, int, float
        The calibrated position data, in the form [x, y, z, ...]
    drift_vel : number or array-like
        The drift velocity in the gas, in cm/us. If scalar, it will only affect the z values. If it's a vector, all
        spatial points will be transformed accordingly.
    clock : int, float
        The CoBo write clock frequency, in MHz

    Returns
    -------
    ndarray or int or float
        The uncalibrated data

    See also
    --------
    pytpc.simulation.drift_velocity_vector

    """
    new_data = data.copy()

    if np.isscalar(drift_vel):
        new_data[:, 2] *= clock / (10 * drift_vel)  # 1 cm/(us.MHz) = 1 cm = 10 mm
    else:
        assert drift_vel.shape == (3,), 'vector drift velocity must have 3 dimensions'
        tbs = data[:, 2] * clock / (10 * -drift_vel[2]) + offset  # 1 cm/(us.MHz) = 1 cm = 10 mm
        new_data[:, 0:3] -= np.outer(tbs, -drift_vel) / clock * 10
        new_data[:, 2] = tbs

    return new_data


def load_pedestals(filename):
    """ Loads pedestals from the provided file.

    The pedestals file should be in CSV format, and the columns should be cobo, asad, aget, and channel.

    Parameters
    ----------
    filename : string
        The name of the file containing the pedestals.

    Returns
    -------
    peds : ndarray
        The pedestals, in a 4D array with the same indices as above.
    """

    f = open(filename, 'r')
    peds = np.zeros((10, 4, 4, 68, 512))

    for line in f:
        subs = line[:-1].split(',')  # throw out the newline and split at commas
        vals = [int(float(x)) for x in subs]
        peds[(vals[0], vals[1], vals[2], vals[3])] = vals[4] * np.ones(512)

    return peds


def load_padmap(filename):
    """Loads pad mapping from the provided file.

    Parameters
    ----------
    filename : string
        The name of the file containing the pad mapping

    Returns
    -------
    pm : dict
        The pad mapping as {(cobo, asad, aget, ch): pad}
    """

    f = open(filename, 'r')
    pm = {}

    for line in f:
        subs = line[:-1].split(',')  # throw out the newline and split at commas
        vals = [int(float(x)) for x in subs]
        pm[(vals[0], vals[1], vals[2], vals[3])] = vals[4]

    return pm


class Error(Exception):
    """Base class for custom exceptions."""
    pass


class FilePosError(Error):
    """Raised when data is read from a file whose read pointer does not seem
    to be in the correct position.

    Parameters
    ----------
    filename : string
        The name of the file
    details : string, optional
        More details about the error
    """

    def __init__(self, filename, details=""):
        self.filename = filename  #: The name of the file where the error occurred
        self.details = details    #: Details of the error
        self.message = "FilePosError in " + filename + ": " + details  # An error message for printing
