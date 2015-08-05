from __future__ import division, print_function

import os
import struct
import numpy as np

class GRAWFile(object):
    def __init__(self, filename):

        self.lookup = []
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

        try:
            self.fp = open(filename, 'rb')
            self.is_open = True
            # print("Opened file: " + filename)

        except IOError:
            raise

        return

    def close(self):
        """Close an open file"""

        if self.is_open:
            self.fp.close()
            self.is_open = False

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
                  'frame_size': self._bsmerge(hdr_raw[1:4]),
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
        self.fp.seek(hdr_start + header['header_size'] * 128)

class GRAWFrame(object):

    def __init__(self, header, data):
        self.header = header
        self.data = data
