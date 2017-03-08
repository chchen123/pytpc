from __future__ import division, print_function
import struct
import numpy as np

import logging
logger = logging.getLogger(__name__)


class BadVMEDataError(Exception):
    pass


class ADCEvent(object):
    def __init__(self, evt_id, timestamp, coincidence_register, data):
        self.evt_id = evt_id
        self.timestamp = timestamp
        self.coincidence_register = np.array([(coincidence_register & 1 << i) > 0 for i in range(16)])
        self.data = data


class ScalerEvent(object):
    def __init__(self, index, scalers):
        self.index = index
        self.scalers = scalers

    def __str__(self):
        return f"Scaler event {self.index:d}: {self.scalers:s}"


class VMEFile(object):

    def __init__(self, filename):
        self.fp = open(filename, 'rb')
        self.adc_events_seen = 0
        self.scaler_events_seen = 0

        self._file_len = self.fp.seek(0, 2)
        self.fp.seek(0)

    def __len__(self):
        """Returns length of file, in bytes."""
        return self._file_len

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
            scalers = np.fromfile(self.fp, dtype='<I', count=18)
            sentinel = struct.unpack('<I', self.fp.read(4))[0]
            if sentinel != 0xffffffff:
                raise BadVMEDataError(f'Invalid sentinel value {sentinel:x} at end of scaler frame')
            return ScalerEvent(
                index=self.scaler_events_seen - 1,
                scalers=scalers,
            )

        elif evthdr == 0x17fb:
            # evtlen = evthdr & 0xfff
            evtnum, timestamp, coinreg = struct.unpack('<III', self.fp.read(12))

            # Registers are evt config reg., address counter, evt counter, addr of end of event
            reg1 = np.fromfile(self.fp, dtype='<I', count=4)
            raw1 = np.fromfile(self.fp, dtype='<I', count=512)

            reg2 = np.fromfile(self.fp, dtype='<I', count=4)
            raw2 = np.fromfile(self.fp, dtype='<I', count=512)

            self.adc_events_seen += 1
            true_evtnum = evtnum - self.scaler_events_seen  # evt num is incremented even for a scaler buffer

            last_tb1 = reg1[3] & 0x1ffff
            last_tb2 = reg2[3] & 0x1ffff
            if last_tb1 != last_tb2:
                logger.warning('Last TBs do not match for event %d (frame 0x%x): %d != %d',
                               true_evtnum, evtnum, last_tb1, last_tb2)

            raw1 = np.roll(raw1, -last_tb1, axis=-1)
            raw2 = np.roll(raw2, -last_tb2, axis=-1)

            adc_data = np.zeros((4, 512), dtype='int32')
            adc_data[0] = (raw1 & 0x1fff0000) >> 16
            adc_data[1] = (raw1 & 0x1fff)
            adc_data[2] = (raw2 & 0x1fff0000) >> 16
            adc_data[3] = (raw2 & 0x1fff)

            return ADCEvent(
                evt_id=true_evtnum,
                timestamp=timestamp,
                coincidence_register=coinreg,
                data=adc_data,
            )
        else:
            raise IOError(f'Invalid header: {evthdr:x}')

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
