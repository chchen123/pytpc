from .multiplicity import MultiplicityTrigger
from pytpc.fitting.mixins import TrackerMixin, EventGeneratorMixin
from pytpc.utilities import read_lookup_table, find_exclusion_region
import os
import logging

__all__ = ['TriggerSimulator']

logger = logging.getLogger(__name__)


class TriggerSimulator(EventGeneratorMixin, TrackerMixin):
    def __init__(self, config, xcfg_path=None):
        super().__init__(config)

        self.padmap = read_lookup_table(config['padmap_path'])  # maps (cobo, asad, aget, ch) -> pad
        self.reverse_padmap = {v: k for k, v in self.padmap.items()}  # maps pad -> (cobo, asad, aget, ch)
        if xcfg_path is not None:
            self.excluded_pads, self.low_gain_pads = find_exclusion_region(xcfg_path, self.padmap)
            logger.info('Excluding pads from regions in file %s', os.path.basename(xcfg_path))
        else:
            self.excluded_pads = []
            self.low_gain_pads = []

        self.badpads = set(self.excluded_pads).union(set(self.low_gain_pads))
        logger.info('%d pads will be excluded', len(self.badpads))

        self.trigger = MultiplicityTrigger(config, self.reverse_padmap)

    def run_track(self, params):
        tr = self.tracker.track_particle(*params)
        evt = self.evtgen.make_event(tr[:, :3], tr[:, 4])

        for k in self.badpads.intersection(evt.keys()):
            del evt[k]

        trig = self.trigger.find_trigger_signals(evt)
        mult = self.trigger.find_multiplicity_signals(trig)
        did_trig = self.trigger.did_trigger(mult)

        result = {
            'num_pads_hit': len(evt),  # evt is a dict of traces indexed by pad number
            'trig': did_trig,
        }

        return result
