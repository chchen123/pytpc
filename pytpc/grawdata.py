from __future__ import division, print_function

import os
import struct
import numpy as np

class GRAWFile(object):

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

        self.fp = open(filename, 'rb')
        self.is_open = True
        # print("Opened file: " + filename)

        return

    def close(self):
        """Close an open file"""

        if self.is_open:
            self.fp.close()
            self.is_open = False

        return

    def make_lookup_table(self):

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
        res = 0
        for i, x in enumerate(a):
            res += x << 8*(len(a) - i - 1)
        return res

    def _read(self):
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

        data_begin = hdr_start + header['header_size'] * 256
        data_end = data_begin + header['num_items'] * header['item_size']

        datatype = np.dtype([('cobo', 'u1'), ('asad', 'u1'), ('aget', 'u1'), ('channel', 'u1'),
                             ('pad', 'u2'), ('data', '512i2')])

        if header['frame_type'] == GRAWFile.partial_readout_frame_type:
            raw = np.fromfile(self.fp, dtype='>I4', count=header['num_items'])
            agets = (raw & 0xC0000000)>>30
            channels = (raw & 0x3F800000)>>23
            tbs = (raw & 0x007FC000)>>14
            samples = (raw & 0x00000FFF)

            ach = np.vstack((agets, channels)).T
            pairs = ach[np.unique(ach)]

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

        return res

    def __getitem__(self, item):

        if not isinstance(item, int):
            raise TypeError('Can only index file with an int')

        if item < 0 or item > len(self.lookup):
            raise IndexError("The index is out of bounds")
        else:
            self.current_event = item
            self.fp.seek(self.lookup[item])
            return self._read()

    def __len__(self):
        return len(self.lookup)
