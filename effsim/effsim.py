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

from .database import ParameterSet, TriggerResult, CleaningResult, MinimizerResult, ClockOffsets
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
    def __init__(self, config, pedestals=None, corrupt_cobo_clocks=False):
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

        self.padmap = read_lookup_table(config['padmap_path'])  # maps (cobo, asad, aget, ch) -> pad
        self.pad_cobos = np.zeros(10240, dtype=np.int64)  # List index = pad, value = cobo
        for (cobo, asad, aget, ch), pad in self.padmap.items():
            self.pad_cobos[pad] = cobo

        self.cobo_offsets = self.make_cobo_clock_offsets()
        if corrupt_cobo_clocks:
            self.apply_cobo_clock_patch()

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

    def make_cobo_clock_offsets(self):
        cobo_offsets = np.zeros(10)
        cobo_offsets[1:] = np.random.triangular(left=-1, mode=0, right=1, size=9)
        return cobo_offsets

    @property
    def pad_offsets(self):
        return self.cobo_offsets[self.pad_cobos]

    def apply_cobo_clock_patch(self):
        logger.warning("Applying patch to Event.xyzs to corrupt CoBo clocks with random offsets")
        old_xyzs = Event.xyzs  # patch in the scope of this module

        def patched_xyzs(self_, *args, **kwargs):
            assert 'return_pads' in kwargs, "Must call with 'return_pads == True' for this hack to work"
            assert len(args) == 0, "I can't work with args"
            assert 'vd' not in kwargs, "We can't calibrate in this step with the patch"

            xyz = old_xyzs(self_, *args, **kwargs)  # format: [x, y, z, a, pad, ...]
            xyz[:, 2] += self.pad_offsets[xyz[:, 4].astype('int')]
            return xyz

        Event.xyzs = patched_xyzs


class EfficiencySimulator(object):
    def __init__(self, config, excluded_pads=[], lowgain_pads=[], evtgen_config=None, pedestals=None,
                 corrupt_cobo_clocks=False):
        if evtgen_config is None:
            evtgen_config = config
        self.evtsim = EventSimulator(evtgen_config, badpads=lowgain_pads)
        self.noisemaker = NoiseMaker(evtgen_config, pedestals=pedestals, corrupt_cobo_clocks=corrupt_cobo_clocks)
        self.trigger = TriggerSimulator(config, excluded_pads=excluded_pads, pedestals=pedestals)
        self.cleaner = EventCleaner(config)
        self.fitter = MCFitter(config)

        self.clocks_are_corrupted = corrupt_cobo_clocks

    def setup_clock_offsets(self, evt_id):
        self.noisemaker.cobo_offsets = self.noisemaker.make_cobo_clock_offsets()

        cobo_names = ('cobo{:d}'.format(i) for i in range(10))
        offset_map = dict(zip(cobo_names, self.noisemaker.cobo_offsets))
        return ClockOffsets(evt_id=evt_id, **offset_map)

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

            if self.clocks_are_corrupted:
                clock_res = self.setup_clock_offsets(evt_id)
                session.add(clock_res)

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
