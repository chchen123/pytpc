from __future__ import division, print_function
import os
import struct
import numpy as np
import pytpc.evtdata
import pytpc.datafile
import logging

logger = logging.getLogger(__name__)

class GRAWFile(pytpc.datafile.DataFile):
    """An object representing an unmerged GRAW file from the DAQ.

    Parameters
    ----------
    filename : string
        The path to the file
    max_len : number
        The maximum number of events to read from the file. Set this to a reasonable value to make the files open
        more quickly.

    """
    full_readout_frame_type = 2
    partial_readout_frame_type = 1

    def __init__(self, filename, open_mode='r', max_len=None):

        self.evtids = None
        self.max_len = max_len

        super().__init__(filename, open_mode)

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

                if self.max_len is not None and len(self.lookup) >= self.max_len:
                    break

        self.fp.seek(0)
        self.evtids = np.array(self.evtids)

        ltfilename = self.fp.name + '.lookup'

        if not os.path.exists(ltfilename):
            ltfile = open(ltfilename, 'w')
            for i, e in zip(self.lookup, self.evtids):
                ltfile.write(str(i) + ',' + str(e) + '\n')
            ltfile.close()

    def load_lookup_table(self, filename):

        self.lookup = []
        self.evtids = []

        try:
            file = open(filename, 'r')
            for line in file:
                l, e = [int(a) for a in line.strip().split(',')]
                self.lookup.append(l)
                self.evtids.append(e)
        except FileNotFoundError:
            print("File name was not valid")
            return

        self.evtids = np.array(self.evtids)

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
        raw = self._read_raw()
        return self._parse(raw, return_header=return_header)

    def _read_raw(self):

        startpos = self.fp.tell()
        self.fp.seek(startpos + 1)
        frame_size = self._bsmerge(self.fp.read(3)) * 256
        self.fp.seek(startpos)
        return self.fp.read(frame_size)

    @staticmethod
    def _unpack_data_partial_readout(raw):
        agets = (raw & 0xC0000000)>>30
        channels = (raw & 0x3F800000)>>23
        tbs = (raw & 0x007FC000)>>14
        samples = (raw & 0x00000FFF)

        return agets, channels, tbs, samples

    @staticmethod
    def _unpack_data_full_readout(raw):
        agets = (raw & 0xC000)>>14
        samples = (raw & 0x0FFF)

        return agets, samples

    @staticmethod
    def _parse(rawframe, return_header=False):

        hdr_raw = struct.unpack('>5BHB2HL6BL2BHB32B4HL4H', rawframe[:83])
        header = {'metatype': hdr_raw[0],
                  'frame_size': GRAWFile._bsmerge(hdr_raw[1:4]) * 256,
                  'data_source': hdr_raw[4],
                  'frame_type': hdr_raw[5],
                  'revision': hdr_raw[6],
                  'header_size': hdr_raw[7],
                  'item_size': hdr_raw[8],
                  'num_items': hdr_raw[9],
                  'timestamp': GRAWFile._bsmerge(hdr_raw[10:16]),
                  'evt_id': hdr_raw[16],
                  'cobo': hdr_raw[17],
                  'asad': hdr_raw[18],
                  'offset': hdr_raw[19],
                  'status': hdr_raw[20]}

        logger.debug('Frame header: metatype %d, type %d, revision %d, with %d items',
                     header['metatype'], header['frame_type'], header['revision'], header['num_items'])

        datatype = np.dtype([('cobo', 'u1'), ('asad', 'u1'), ('aget', 'u1'), ('channel', 'u1'),
                             ('pad', 'u2'), ('data', '512i2')])

        datamin = header['header_size'] * 256

        log_frameid = '{evt_id}:{cobo}/{asad}'.format(**header)

        if header['frame_type'] == GRAWFile.partial_readout_frame_type:
            raw = np.frombuffer(rawframe[datamin:], dtype='>u4', count=header['num_items'])
            agets, channels, tbs, samples = GRAWFile._unpack_data_partial_readout(raw)

            if agets.min() < 0 or agets.max() > 3:
                logger.warn('(%s) Invalid AGET indices present: min=%d, max=%d', log_frameid, agets.min(), agets.max())
            if channels.min() < 0 or channels.max() > 67:
                logger.warn('(%s) Invalid channels present: min=%d, max=%d', log_frameid, channels.min(), channels.max())
            if tbs.min() < 0 or tbs.max() > 511:
                logger.warn('(%s) Invalid time buckets present: min=%d, max=%d', log_frameid, tbs.min(), tbs.max())
            if samples.min() < 0 or samples.max() > 4095:
                logger.warn('(%s) Invalid samples present: min=%d, max=%d', log_frameid, samples.min(), samples.max())

            ach = np.column_stack((agets, channels))
            pairs = np.unique(ach.view(dtype=[('f0', 'u4'), ('f1', 'u4')])).view(dtype='u4').reshape(-1, 2)

            res = np.zeros(len(pairs), dtype=datatype)

            for i, (aget, ch) in enumerate(pairs):
                idx = np.where(np.logical_and(agets == aget, channels == ch))
                t = tbs[idx]
                s = samples[idx]

                assert len(t) == len(s), 'samples and tbs don\'t have same lengths'

                res['cobo'][i] = header['cobo']
                res['asad'][i] = header['asad']
                res['aget'][i] = aget
                res['channel'][i] = ch
                try:
                    res['data'][i, t] = s
                except IndexError:
                    logger.error('(%s) Indexing samples array failed. len(s)=%d, min(t)=%d, max(t)=%d',
                                 log_frameid, len(s), t.min(), t.max())

        elif header['frame_type'] == GRAWFile.full_readout_frame_type:
            raw = np.frombuffer(rawframe[datamin:], dtype='>u2', count=header['num_items'])
            agets, samples = GRAWFile._unpack_data_full_readout(raw)

            if agets.min() < 0 or agets.max() > 3:
                logger.warn('(%s) Invalid AGET indices present: min=%d, max=%d', log_frameid, agets.min(), agets.max())
            if samples.min() < 0 or samples.max() > 4095:
                logger.warn('(%s) Invalid samples present: min=%d, max=%d', log_frameid, samples.min(), samples.max())

            res = np.zeros(68*4, dtype=datatype)

            for aget in np.unique(agets):
                s = samples[np.where(agets == aget)].reshape((512, 68)).T
                idx = slice(aget*68, (aget+1)*68)
                res['cobo'][idx] = header['cobo']
                res['asad'][idx] = header['asad']
                res['aget'][idx] = aget
                res['channel'][idx] = np.arange(68)
                res['data'][idx] = s

        else:
            raise IOError('Invalid frame type: %d' % header['frame_type'])

        if return_header:
            return res, header
        else:
            return res

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

    def get_raw_frames_for_event(self, evtid):
        assert len(self.lookup) == len(self.evtids)

        idx = np.where(self.evtids == evtid)[0]
        res = []
        for i in idx:
            self.fp.seek(self.lookup[i])
            res.append(self._read_raw())
        return res


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
