from .multiplicity import MultiplicityTrigger
from pytpc.utilities import read_lookup_table
import numpy as np
import logging

__all__ = ['TriggerSimulator']

logger = logging.getLogger(__name__)


class TriggerSimulator(object):
    def __init__(self, config, excluded_pads=[], pedestals=None):
        self.padmap = read_lookup_table(config['padmap_path'])  # maps (cobo, asad, aget, ch) -> pad
        self.reverse_padmap = {v: k for k, v in self.padmap.items()}  # maps pad -> (cobo, asad, aget, ch)
        self.badpads = set(excluded_pads)
        logger.info('%d pads will be excluded from trigger', len(self.badpads))

        if pedestals is None:
            pedestals = np.zeros(10240, dtype='float64')
        self.pedestals = pedestals

        self.trigger = MultiplicityTrigger(config, self.reverse_padmap)

    def process_event(self, evt):
        # Remove pads in exclusion region
        non_excluded_pads = set(evt.keys()) - self.badpads
        evt_for_trigger = {k: evt[k] - self.pedestals[k] for k in non_excluded_pads}

        trig, hitmask = self.trigger.find_trigger_signals(evt_for_trigger)
        mult = self.trigger.find_multiplicity_signals(trig)
        did_trig = self.trigger.did_trigger(mult)

        return did_trig, hitmask
