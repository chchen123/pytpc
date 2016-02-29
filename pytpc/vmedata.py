from __future__ import division, print_function
import struct
import numpy as np


class VMEFile(object):

    def __init__(self, filename):
        self.fp = open(filename, 'rb')
        self.adc_events_seen = 0
        self.scaler_events_seen = 0

    def _read(self):
        # Find the next event
        while True:
            raw_magic = self.fp.read(2)
            if raw_magic == b'':
                raise EOFError('Reached end of VME file')
            magic = struct.unpack('<H', raw_magic)[0]
            if magic == 0xe238:
                break

        self.fp.seek(-4, 1)  # rewind to previous word, which identifies event type and length
        evthdr, magic = struct.unpack('<HH', self.fp.read(4))
        assert magic == 0xe238, 'magic is now wrong. Out of alignment?'

        if evthdr == 0x2025:
            # This is a scalar event
            self.scaler_events_seen += 1
            return {'type': 'scaler'}
        elif evthdr == 0x17fb:
            # evtlen = evthdr & 0xfff
            evtnum, timestamp, coinreg = struct.unpack('<III', self.fp.read(12))

            # Registers are evt config reg., address counter, evt counter, addr of end of event
            reg1 = np.fromfile(self.fp, dtype='<I', count=4)
            raw1 = np.fromfile(self.fp, dtype='<I', count=512)

            reg2 = np.fromfile(self.fp, dtype='<I', count=4)
            raw2 = np.fromfile(self.fp, dtype='<I', count=512)

            adc_data = np.zeros((3, 512), dtype='int32')
            adc_data[0] = (raw1 & 0x1fff0000) >> 16
            adc_data[1] = (raw1 & 0x1fff)
            adc_data[2] = (raw2 & 0x1fff0000) >> 16

            self.adc_events_seen += 1
            true_evtnum = evtnum - self.scaler_events_seen  # evt num is incremented even for a scaler buffer

            last_tb1 = reg1[3] & 0x1ffff
            last_tb2 = reg2[3] & 0x1ffff
            assert last_tb1 == last_tb2, 'last TBs did not match'

            evtdict = {'type': 'adc',
                       'evt_num': true_evtnum,
                       'timestamp': timestamp,
                       'coin_reg': coinreg,
                       'last_tb': last_tb1,
                       'adc_data': adc_data}
            return evtdict
        else:
            raise IOError('Invalid header:', hex(evthdr))

    def __iter__(self):
        self.fp.seek(0)
        self.scaler_events_seen = 0
        self.adc_events_seen = 0
        return self

    def __next__(self):
        while True:
            try:
                evt = self._read()
                return evt
            except EOFError as e:
                raise StopIteration from e
            except IOError:
                continue
