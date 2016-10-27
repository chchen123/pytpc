import numpy as np
import pytpc
from pytpc.evtdata import Event
from pytpc.utilities import read_lookup_table
from pytpc.fitting import MCFitter, BadEventError
from pytpc.fitting.mixins import TrackerMixin, EventGeneratorMixin
from pytpc.cleaning import EventCleaner, apply_clean_cut
from pytpc.trigger import TriggerSimulator
from pytpc.constants import pi
from pytpc.padplane import generate_pad_plane
import logging
from copy import copy

from .database import ParameterSet, TriggerResult, CleaningResult, MinimizerResult
from .database import EventCannotContinue, managed_session

logger = logging.getLogger(__name__)


def three_point_center(p1, p2, p3):
    mp1 = (p1 + p2) / 2
    mp2 = (p2 + p3) / 2
    sl1 = (p1[0] - p2[0]) / (p2[1] - p1[1])
    sl2 = (p2[0] - p3[0]) / (p3[1] - p2[1])
    xc = (sl1 * mp1[0] - sl2 * mp2[0] - mp1[1] + mp2[1]) / (sl1 - sl2)
    yc = (-sl2 * mp1[1] + sl1 * (sl2 * (mp1[0] - mp2[0]) + mp2[1])) / (sl1 - sl2)
    return xc, yc


class EventSimulator(TrackerMixin, EventGeneratorMixin):
    def __init__(self, config, badpads=[]):
        super().__init__(config)
        self.untiltmat = pytpc.utilities.tilt_matrix(self.tilt)
        self.badpads = set(badpads)
        logger.info('%d pads will be dropped from the generated events', len(badpads))
        self.padmap = read_lookup_table(config['padmap_path'])  # maps (cobo, asad, aget, ch) -> pad
        self.reverse_padmap = {v: k for k, v in self.padmap.items()}  # maps pad -> (cobo, asad, aget, ch)
        self.noise_stddev = config['noise_stddev']

    def make_event(self, x0, y0, z0, enu0, azi0, pol0):
        tr = self.tracker.track_particle(x0, y0, z0, enu0, azi0, pol0)
        tr = tr[np.where(np.logical_and(tr[:, 2] > 0, tr[:, 2] < 1))].copy()
        pos = tr[:, :3]
        en = tr[:, 4]

        cx, cy = three_point_center(tr[0, :2], tr[1, :2], tr[2, :2])
        center = np.array([cx, cy, 0]) * 1000
        center = self.untiltmat @ center
        center[1] -= np.tan(self.tilt) * 1000
        center = pytpc.evtdata.uncalibrate(center.reshape((-1, 3)), self.vd, self.clock).ravel()

        evt = self.evtgen.make_event(pos, en)

        hitpads = set(evt.keys())
        for p in self.badpads.intersection(hitpads):
            del evt[p]

        return evt, center[:2]

    def convert_event(self, dict_evt, evt_id=0, timestamp=0):
        py_evt = Event(evt_id, timestamp)
        py_evt.traces = np.zeros(len(dict_evt), dtype=py_evt.dt)

        for i, (pad, data) in enumerate(dict_evt.items()):
            tr_item = py_evt.traces[i]
            tr_item['cobo'], tr_item['asad'], tr_item['aget'], tr_item['channel'] = self.reverse_padmap[pad]
            tr_item['pad'] = pad
            tr_item['data'] = data

        return py_evt


class NoiseMaker(object):
    def __init__(self, config, pedestals=None):
        config = copy(config)
        config['mass_num'] = config['beam_mass']
        config['charge_num'] = config['beam_charge']
        config['tracker_max_en'] = config['beam_enu0'] * config['beam_mass'] * 1.1  # The last part is just to be safe

        self.beam_sim = EventSimulator(config)

        beam_evt, _ = self.beam_sim.make_event(0, 0, 1.0, config['beam_enu0'], 0, pi)
        self.beam_mesh = sum((v for v in beam_evt.values()))
        self.beam_mesh /= self.beam_mesh.max()

        pads = generate_pad_plane()
        self.bigpads = np.where(np.round(np.abs(pads[:, 1, 1] - pads[:, 0, 1])) > 6)[0]
        self.smallpads = np.where(np.round(np.abs(pads[:, 1, 1] - pads[:, 0, 1])) < 6)[0]
        assert(len(self.bigpads) + len(self.smallpads) == 10240)

        self.baseline_depression_scale = config['baseline_depression_scale']
        self.big_pad_multiplier = config['big_pad_multiplier']

        self.noise_stddev = config['noise_stddev']

        if pedestals is not None:
            self.pedestals = pedestals
        else:
            self.pedestals = np.zeros(10240, dtype='float64')

    def add_noise(self, evt, depress_baseline=True, add_gaussian_noise=True, clip=True):
        for pad, trace in evt.items():
            if add_gaussian_noise:
                trace += np.random.normal(self.pedestals[pad], self.noise_stddev, len(trace))

            if depress_baseline:
                mult = self.big_pad_multiplier if pad in self.bigpads else 1
                trace -= self.beam_mesh * self.baseline_depression_scale * mult

            if clip:
                np.clip(trace, a_min=0, a_max=4095, out=trace)  # Perform clip in-place

        return evt


class EfficiencySimulator(object):
    def __init__(self, config, excluded_pads=[], lowgain_pads=[], evtgen_config=None, pedestals=None):
        if evtgen_config is None:
            evtgen_config = config
        self.evtsim = EventSimulator(evtgen_config, badpads=lowgain_pads)
        self.noisemaker = NoiseMaker(evtgen_config, pedestals=pedestals)
        self.trigger = TriggerSimulator(config, excluded_pads=excluded_pads, pedestals=pedestals)
        self.cleaner = EventCleaner(config)
        self.fitter = MCFitter(config)

    def make_event(self, evt_id, params):
        dict_evt, true_ctr = self.evtsim.make_event(*params)

        return dict_evt, true_ctr

    def prepare_event_for_cleaner(self, dict_evt, hitmask):
        sliced_evt = {k: v for k, v in dict_evt.items() if bool(hitmask[k])}
        return self.evtsim.convert_event(sliced_evt)

    def run_trigger(self, evt_id, dict_evt):
        didtrig, hitmask = self.trigger.process_event(dict_evt)
        num_hits = len(hitmask.nonzero()[0])
        return TriggerResult(evt_id=evt_id, did_trigger=didtrig, num_pads_hit=num_hits), hitmask

    def run_cleaner(self, evt_id, evt):
        clean_xyz_full, ctr = self.cleaner.process_event(evt)

        clean_xyz = apply_clean_cut(clean_xyz_full)
        num_pts_before = len(clean_xyz_full)
        num_pts_after = len(clean_xyz)

        clean_res = CleaningResult(evt_id=evt_id, num_pts_before=num_pts_before, num_pts_after=num_pts_after)

        return clean_res, clean_xyz, ctr

    def run_fit(self, evt_id, clean_xyz, ctr):
        mcres = self.fitter.process_event(clean_xyz, ctr[0], ctr[1], preprocess_kwargs={'rotate_pads': False})
        mcres['evt_id'] = int(evt_id)
        return MinimizerResult(**mcres)

    def process_event(self, evt_id, param_vector):
        with managed_session() as session:
            param_set = ParameterSet(
                evt_id=evt_id,
                x0=param_vector[0],
                y0=param_vector[1],
                z0=param_vector[2],
                enu0=param_vector[3],
                azi0=param_vector[4],
                pol0=param_vector[5],
            )
            session.add(param_set)

            try:
                dict_evt, true_ctr = self.make_event(evt_id, param_vector)
            except Exception as err:
                raise EventCannotContinue('Simulation failed for event {:d}'.format(evt_id)) from err

            try:
                dict_evt = self.noisemaker.add_noise(dict_evt)
            except Exception as err:
                raise EventCannotContinue('Failed to add noise for event {:d}'.format(evt_id)) from err

            try:
                trig_res, hitmask = self.run_trigger(evt_id, dict_evt)
            except Exception as err:
                raise EventCannotContinue('Trigger failed for event {:d}'.format(evt_id)) from err
            else:
                session.add(trig_res)

            try:
                evt = self.prepare_event_for_cleaner(dict_evt, hitmask)
                clean_res, clean_xyz, ctr = self.run_cleaner(evt_id, evt)
            except Exception as err:
                raise EventCannotContinue('Cleaning failed for event {:d}'.format(evt_id)) from err
            else:
                session.add(clean_res)

            try:
                fit_res = self.run_fit(evt_id, clean_xyz, ctr)
            except BadEventError as err:
                logger.warning('Fit failed for event %d: %s', evt_id, str(err))
            except Exception as err:
                raise EventCannotContinue('Fit failed for event {:d}'.format(evt_id)) from err
            else:
                session.add(fit_res)
