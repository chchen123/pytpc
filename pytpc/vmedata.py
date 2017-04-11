from __future__ import division, print_function
import struct
import numpy as np
import h5py

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

            # Check that the frame index is consistent
            if (evtnum != self.adc_events_seen + self.scaler_events_seen - 1):
                raise BadVMEDataError(f'Frame index is inconsistent at frame index {evtnum:x}')

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


class VMEAlignmentTable:
    """A table that maps VME event IDs to GET electronics event IDs.
    
    This keeps track of the results of this alignment. It maintains a list of offsets
    between the two DAQs for each event and a list of which events have been invalidated
    because they are misaligned and not recoverable.
    
    Parameters
    ----------
    num_events : int
        The number of events in the table. This should correspond to the number of events
        in the VME DAQ.
    
    """

    # Identifiers for the output HDF5 file
    hdf_group_name = 'alignment'
    hdf_offset_ds_name = 'offsets'
    hdf_valid_ds_name = 'valid'

    def __init__(self, num_events):
        # Offsets are defined such that VME + offset = GET.
        # Thus, they are indexed by VME event ID
        self.offsets = np.zeros(num_events, dtype='int')
        self.valid = np.full(num_events, True, dtype=bool)

    def vme_to_get(self, evt_id):
        """Convert a VME event ID to the corresponding GET event ID.
        
        This applies the correct offset to the VME event ID to get the GET event ID.
        
        Parameters
        ----------
        evt_id : int
            The VME event ID.

        Returns
        -------
        int
            The GET event ID.

        """
        return evt_id + self.offsets[evt_id]

    def update_offset(self, start_vme_evt_id, offset):
        """Increment the offset for all events after the given VME event ID by the given offset. 
        
        Parameters
        ----------
        start_vme_evt_id : int
            The first event ID whose offset will be updated.
        offset : int
            The stored offsets for all events after the first one given will be increased by
            this amount. Use a negative value to decrease the offsets.

        """
        self.offsets[start_vme_evt_id:] += offset

    def invalidate_range(self, begin, end):
        """Mark events between ``begin`` and ``end`` as invalid.
        
        The upper bound ``end`` is not included.
        
        Parameters
        ----------
        begin, end : int
            The first and last events to mark. The last event is not included.

        """
        self.valid[begin:end] = False
        logger.warning(f'VME events {begin} through {end} were invalidated')

    def to_hdf(self, path):
        """Write the results to an HDF5 file.
        
        Parameters
        ----------
        path : str
            Path to output file.

        """
        with h5py.File(path) as hf:
            group = hf.create_group(self.hdf_group_name)
            group.create_dataset(self.hdf_offset_ds_name, data=self.offsets)
            group.create_dataset(self.hdf_valid_ds_name, data=self.valid)

    @classmethod
    def from_hdf(cls, path):
        """Load the table from the given HDF5 file.
        
        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        VMEAlignmentTable
            An instance of this class containing the data from the file.

        """
        with h5py.File(path, 'r') as hf:
            group = hf[cls.hdf_group_name]
            offsets = group[cls.hdf_offset_ds_name][:]
            valid = group[cls.hdf_valid_ds_name][:]

        instance = cls(len(offsets))
        instance.offsets = offsets
        instance.valid = valid
        return instance
