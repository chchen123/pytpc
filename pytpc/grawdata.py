from __future__ import division, print_function
import os
import struct
import numpy as np
import pytpc.evtdata


class GRAWFile(object):
    """An object representing an unmerged GRAW file from the DAQ.

    Parameters
    ----------
    filename : string
        The path to the file

    """
    full_readout_frame_type = 2
    partial_readout_frame_type = 1

    def __init__(self, filename):
        self.lookup = []
        self.evtids = None
        """A lookup table for the events in the file. This is simply an array of file offsets."""

        self.current_event = 0  #: The index of the current event

        if filename is not None:
            self.open(filename)
            ltfilename = os.path.splitext(filename)[0] + '.lookup'
            if os.path.exists(ltfilename):
                self.load_lookup_table(ltfilename)
            else:
                self.make_lookup_table()
            self.is_open = True

        else:
            self.fp = None
            self.is_open = False

        return

    def open(self, filename):
        """Open the given file.

        When a file is open, the attribute `is_open` should be True.

        Parameters
        ----------
        filename : string
            The file to open
        """
        self.fp = open(filename, 'rb')
        self.is_open = True
        return

    def close(self):
        """Close an open file"""
        if self.is_open:
            self.fp.close()
            self.is_open = False
        return

    def make_lookup_table(self):
        """Make the lookup table for the file by indexing its contents.

        This finds the offset at which each frame begins in the file. These are stored in
        the attribute `lookup`.

        Another list, `evtids`, is also filled. This contains the event ID for each frame.
        """
        self.lookup = []
        self.evtids = []
        self.fp.seek(0)

        metatype = self.fp.read(1)
        self.fp.seek(0)
        while True:
            pos = self.fp.tell()
            mt = self.fp.read(1)
            if mt == b'':
                break
            elif mt != metatype:
                raise IOError('Bad metatype: file out of position?')
            else:
                self.lookup.append(pos)
                size = self._bsmerge(self.fp.read(3)) * 256
                self.fp.seek(pos + 22)
                evtid = struct.unpack('>L', self.fp.read(4))[0]
                self.evtids.append(evtid)
                self.fp.seek(pos + size)

        self.fp.seek(0)
        self.evtids = np.array(self.evtids)

    def load_lookup_table(self, filename):
        """Read a lookup table from a file.

        The file should contain the (integer) file offsets, with one on each line.

        Parameters
        ----------
        filename : string
            The path to the file.
        """
        self.lookup = []
        try:
            file = open(filename, 'r')
            for line in file:
                self.lookup.append(int(line.rstrip()))
        except FileNotFoundError:
            print("File name was not valid")
            return

    @staticmethod
    def _bsmerge(a):
        """Byte-swap and concatenate the bytes in the given iterable.

        Parameters
        ----------
        a : iterable of bytes
            The bytes

        Returns
        -------
        res : integer
            The byte-swapped number
        """
        res = 0
        for i, x in enumerate(a):
            res += x << 8*(len(a) - i - 1)
        return res

    def _read(self, return_header=False):
        """Read the frame beginning at the current location of the file cursor.

        This should probably not be used directly. Subscript the file instead.

        Parameters
        ----------
        return_header : bool
            If true, return the header as well

        Results
        -------
        res : ndarray
            The data in the frame
        header : dict
            The header information. Returned only if `return_header` is True.

        Raises
        ------
        IOError
            If the frame type given in the header is not recognized.
        """
        hdr_start = self.fp.tell()
        hdr_raw = struct.unpack('>5BHB2HL6BL2BHB32B4HL4H', self.fp.read(83))
        header = {'metatype': hdr_raw[0],
                  'frame_size': self._bsmerge(hdr_raw[1:4]) * 256,
                  'data_source': hdr_raw[4],
                  'frame_type': hdr_raw[5],
                  'revision': hdr_raw[6],
                  'header_size': hdr_raw[7],
                  'item_size': hdr_raw[8],
                  'num_items': hdr_raw[9],
                  'timestamp': self._bsmerge(hdr_raw[10:16]),
                  'evt_id': hdr_raw[16],
                  'cobo': hdr_raw[17],
                  'asad': hdr_raw[18],
                  'offset': hdr_raw[19],
                  'status': hdr_raw[20]}
        self.fp.seek(hdr_start + header['header_size'] * 256)

        datatype = np.dtype([('cobo', 'u1'), ('asad', 'u1'), ('aget', 'u1'), ('channel', 'u1'),
                             ('pad', 'u2'), ('data', '512i2')])

        if header['frame_type'] == GRAWFile.partial_readout_frame_type:
            raw = np.fromfile(self.fp, dtype='>I4', count=header['num_items'])
            agets = (raw & 0xC0000000)>>30
            channels = (raw & 0x3F800000)>>23
            tbs = (raw & 0x007FC000)>>14
            samples = (raw & 0x00000FFF)

            ach = np.vstack((agets, channels)).T
            pairs = {tuple(a) for a in ach}

            res = np.zeros(len(pairs), dtype=datatype)

            for i, (aget, ch) in enumerate(pairs):
                idx = np.where(np.logical_and(agets == aget, channels == ch))
                t = tbs[idx]
                s = samples[idx]

                assert len(t) == len(s), 'samples and tbs don\'t have same lengths'
                assert tbs.max() <= len(s), 'max tb outside range of sample array length'

                res[i] = header['cobo'], header['asad'], aget, ch, 0, s[t]

        elif header['frame_type'] == GRAWFile.full_readout_frame_type:
            raw = np.fromfile(self.fp, dtype='>I2', count=header['num_items'])
            agets = (raw & 0xC000)>>14
            samples = (raw & 0x0FFF)

            res = np.zeros(68*4, dtype=datatype)

            for aget in agets[np.unique(agets)]:
                s = samples[np.where(agets == aget)].reshape((512, 68)).T
                idx = slice(aget*68, aget*(68+1))
                res[idx]['cobo'] = header['cobo']
                res[idx]['asad'] = header['asad']
                res[idx]['aget'] = aget
                res[idx]['channel'] = np.arange(68)
                res[idx]['data'] = s

        else:
            raise IOError('Invalid frame type: %d' % header['frame_type'])

        if self.fp.tell() != hdr_start + header['frame_size']:
            print('Frame had zero padding at end')
            self.fp.seek(hdr_start + header['frame_size'])

        if return_header:
            return res, header
        else:
            return res

    def __getitem__(self, item):
        if item < 0 or item > len(self.lookup):
            raise IndexError("The index is out of bounds")
        else:
            self.current_event = item
            self.fp.seek(self.lookup[item])
            return self._read()

    def __len__(self):
        return len(self.lookup)

    def __str__(self):
        return 'GRAW file with path %s' % os.path.basename(self.fp.name)

    def __repr__(self):
        return 'GRAWFile(%s)' % os.path.basename(self.fp.name)

    def _frame_generator(self, a):
        """Return a generator that produces each frame identified by ID in `a`."""
        for i in a:
            yield self[i]

    def get_frames_for_event(self, evtid):
        """Get the frames for the given event ID.

        Each frame with the given event ID is returned by the resulting generator.

        Parameters
        ----------
        evtid : integer
            The event ID to get frames for

        Returns
        -------
        generator
            The frames, as a generator
        """
        assert len(self.lookup) == len(self.evtids)

        idx = np.where(self.evtids == evtid)[0]
        return self._frame_generator(idx)


def merge_frames(files, evtid):
    """Merge the frames for the given event ID from each file into an Event.

    The frames from each file are collected using `GRAWFile.get_frames_for_event`, and are then
    merged together and put into a `pytpc.evtdata.Event` object for further processing.

    Parameters
    ----------
    evtid : integer
        The event ID

    Returns
    -------
    evt : pytpc.evtdata.Event
        An event object. This is the same type that would be returned when reading an event
        from a merged event file.
    """
    frames = []
    for f in files:
        # print('Reading from', f)
        this_frames = list(f.get_frames_for_event(evtid))
        # print('  found', len(this_frames), 'frames')
        frames += this_frames
        # print('  len(frames) is now', len(frames))
    evt = pytpc.evtdata.Event(evtid)
    evt.traces = np.concatenate(frames)

    return evt
