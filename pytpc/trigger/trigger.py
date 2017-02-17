"""trigger.py

Provides TriggerSimulator class for trigger simulation. This wraps a Cython implementation of
a multiplicity trigger.

"""

from .multiplicity import MultiplicityTrigger
from pytpc.utilities import read_lookup_table
import numpy as np
import logging

__all__ = ['TriggerSimulator']

logger = logging.getLogger(__name__)


class TriggerSimulator(object):
    """A simulated trigger for the AT-TPC.

    This class provides a ``process_event`` method that will take an event and determine
    whether the event would have triggered the detector. The trigger condition simulated is a set
    of local (CoBo-level) multiplicity triggers, as in the 46Ar commissioning experiment.
    This does **not** include a MuTAnT.

    Parameters
    ----------
    config : dict
        The analysis config dictionary.
    excluded_pads : iterable, optional
        The set of pads to exclude from the trigger. This is internally converted to a ``set``, so any iterable
        should work. The default value is the empty list ``[]``.
    pedestals : array-like, optional
        An array of the pedestal values indexed by pad number. If this isn't provided, the default is
        zero for each pad.

    """
    def __init__(self, config, excluded_pads=[], pedestals=None):
        #: A lookup table mapping ``(cobo, asad, aget, channel)`` to pad number
        self.padmap = read_lookup_table(config['padmap_path'])

        #: The inverse of ``self.padmap``: maps pad number to ``(cobo, asad, aget, channel)``
        self.reverse_padmap = {v: k for k, v in self.padmap.items()}

        #: The set of pads to exclude from the trigger
        self.badpads = set(excluded_pads)
        logger.info('%d pads will be excluded from trigger', len(self.badpads))

        if pedestals is None:
            pedestals = np.zeros(10240, dtype='float64')
        self.pedestals = pedestals  #: The pedestal values

        self.trigger = MultiplicityTrigger(config, self.reverse_padmap)

    def process_event(self, evt):
        """Determine if the given event would trigger the detector.

        Parameters
        ----------
        evt : dict
            A dictionary mapping pad number to traces as NumPy arrays.

        Returns
        -------
        did_trig : bool
            Whether the event would have triggered the detector.
        hitmask : np.ndarray
            An array of 10 values showing whether each CoBo was hit. The position in the array is the CoBo
            number, and the value is 1 if hit (0 otherwise).

        """
        # Remove pads in exclusion region
        non_excluded_pads = set(evt.keys()) - self.badpads
        evt_for_trigger = {k: evt[k] - self.pedestals[k] for k in non_excluded_pads}

        trig, hitmask = self.trigger.find_trigger_signals(evt_for_trigger)
        mult = self.trigger.find_multiplicity_signals(trig)
        did_trig = self.trigger.did_trigger(mult)

        return did_trig, hitmask
