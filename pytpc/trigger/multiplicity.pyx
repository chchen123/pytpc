"""multiplicity.pyx

This module contains the implementation of a simulated multiplicity trigger for the AT-TPC.

"""

cimport numpy as np
import numpy as np
from libc.math cimport round


cdef _calculate_pad_threshold(int pad_thresh_LSB, int pad_thresh_MSB, double trig_discr_frac):
    """Calculate the pad threshold value in ADC time buckets.

    This transforms the value from what's written in the config file to a more useful value.

    Parameters
    ----------
    pad_thresh_LSB, pad_thresh_MSB : int
        The least-significant and most-significant portions of the pad threshold from the config file.
    trig_discr_frac : double
        The fraction of the ADC's output range that the trigger discriminator covers.

    Returns
    -------
    double
        The threshold in ADC time buckets.

    """
    cdef double pt = ((pad_thresh_MSB << 4) + pad_thresh_LSB)
    cdef double discrMax = trig_discr_frac * 4096  # in data ADC bins
    return (pt / 128) * discrMax


cdef class MultiplicityTrigger:
    """A simulated multiplicity trigger for the AT-TPC.

    This Cython class does the work of checking pad thresholds, simulating the trigger signals,
    and creating the multiplicity signals for the CoBos. It can then determine if the given event
    would have triggered the DAQ.

    Parameters
    ----------
    config : dict
        The analysis config file.
    padmap : dict
        A dictionary mapping pad number to tuples (cobo, asad, aget, channel).

    """

    cdef:
        readonly double write_clock
        """The CoBo write clock, in MHz."""

        readonly double read_clock
        """The CoBo read clock, in MHz."""

        readonly double master_clock
        """The CoBo global master clock, in MHz."""

        readonly double pad_threshold
        """The pad threshold, in ADC time buckets."""

        readonly int trigger_width
        """The width of the trigger signal pulse."""

        readonly double trigger_height
        """The height of the trigger signal pulse."""

        readonly int multiplicity_threshold
        """The multiplicity threshold value from the config file."""

        readonly int multiplicity_window
        """The width of the multiplicity integration window, as listed in the config file."""

        dict padmap

    def __cinit__(self, dict config, dict padmap):
        self.write_clock = float(config['clock']) * 1e6
        self.read_clock = 25e6
        self.master_clock = 100e6

        cdef int pad_thresh_MSB = int(config['pad_thresh_MSB'])
        cdef int pad_thresh_LSB = int(config['pad_thresh_LSB'])
        cdef double trig_discr_frac = float(config['trigger_discriminator_fraction'])
        self.pad_threshold = _calculate_pad_threshold(pad_thresh_LSB, pad_thresh_MSB, trig_discr_frac)

        self.trigger_width = <int> round(float(config['trigger_signal_width']) * self.write_clock)
        self.trigger_height = 48

        self.multiplicity_threshold = config['multiplicity_threshold']
        self.multiplicity_window = <int> round(config['multiplicity_window'] / self.master_clock * self.write_clock)

        self.padmap = padmap

    def find_trigger_signals(self, dict evt):
        """Calculate the trigger signals for the given event.

        Parameters
        ----------
        evt : dict
            A dictionary mapping pad number to traces as numpy arrays.

        Returns
        -------
        result : np.ndarray
            The trigger signals. This has dimensions (10, 512), with CoBo number along the axis 0 and
            sample number along axis 1.
        hitmask : np.ndarray
            An array of 10 values showing whether each CoBo was hit. The position in the array is the CoBo
            number, and the value is 1 if hit (0 otherwise).

        """
        cdef np.ndarray[np.double_t, ndim=2] result = np.zeros((10, 512), dtype=np.double)
        cdef np.ndarray[np.int8_t, ndim=1] hitmask = np.zeros(10240, dtype=np.int8)

        cdef np.ndarray[np.double_t, ndim=1] trigSignal
        cdef int tbIdx, sqIdx
        cdef int padnum
        cdef np.ndarray[np.double_t, ndim=1] padsig
        cdef int cobo

        for padnum, padsig in evt.items():

            trigSignal = np.zeros(512, dtype=np.double)

            tbIdx = 0
            while tbIdx < 512:
                if padsig[tbIdx] > self.pad_threshold:
                    trigSignal[tbIdx:min(tbIdx + self.trigger_width, 512)] += self.trigger_height
                    tbIdx += self.trigger_width
                else:
                    tbIdx += 1

            if np.any(trigSignal > 0):
                cobo = self.padmap[padnum][0]
                result[cobo] += trigSignal
                hitmask[padnum] = True

        return result, hitmask

    def find_multiplicity_signals(self, np.ndarray[np.double_t, ndim=2] trig):
        """Calculate the multiplicity signals from the trigger signals.

        Parameters
        ----------
        trig : np.ndarray
            The trigger signals, as calculated by :meth:`find_trigger_signals`.

        Returns
        -------
        result : np.ndarray
            The multiplicity signals. This array has the same dimensions as the input array.

        """
        cdef np.ndarray[np.double_t, ndim=2] result = np.zeros_like(trig, dtype=np.double)

        cdef int j, minIdx, maxIdx
        cdef double accum
        cdef double time_factor = self.read_clock / self.write_clock

        for j in range(trig.shape[1]):
            minIdx = max(0, j - self.multiplicity_window)
            maxIdx = j

            for i in range(trig.shape[0]):
                accum = 0
                for k in range(minIdx, maxIdx):
                    accum += trig[i, k]
                result[i, j] = accum * time_factor

        return result

    def did_trigger(self, np.ndarray[np.double_t, ndim=2] mult):
        """Determines if the given multiplicity signals rose above the multiplicity threshold in any CoBo.

        Parameters
        ----------
        mult : np.ndarray
            The multiplicity signals, as calculated by :meth:`find_multiplicity_signals`.

        Returns
        -------
        bool
            True if the signal in any CoBo rose above the threshold at any point.

        """
        return np.any(np.max(mult, axis=1) > self.multiplicity_threshold)
