import numpy as np
import pytpc
from pytpc.evtdata import Event
from pytpc.constants import pi, degrees, p_mc2
from pytpc.utilities import find_vertex_energy, read_lookup_table
from pytpc.fitting import MCFitter, BadEventError
from pytpc.fitting.mixins import TrackerMixin, EventGeneratorMixin
from pytpc.relativity import find_proton_params
from pytpc.cleaning import EventCleaner, apply_clean_cut
from pytpc.trigger import TriggerSimulator
import logging
import random

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


def param_generator(beam_enu0, beam_mass, beam_chg, proj_mass, proj_chg, gas, num_evts):
    num_good = 0
    while num_good < num_evts:
        x0 = random.gauss(0, 0.010)
        y0 = random.gauss(0, 0.010)
        z0 = random.uniform(0, 1)
        azi0 = random.uniform(0, 2 * pi)
        pol0 = random.uniform(pi / 2, pi - 10 * degrees)

        vert_en = find_vertex_energy(z0, beam_enu0, beam_mass, beam_chg, gas)  # the total kinetic energy
        if vert_en > beam_enu0 * beam_mass:
            vert_en = 0.0

        _, proj_total_en = find_proton_params(
            th3=pi - pol0,
            m1=beam_mass * p_mc2,
            m2=proj_mass * p_mc2,
            m3=proj_mass * p_mc2,
            m4=beam_mass * p_mc2,
            T=vert_en
        )
        enu0 = proj_total_en - proj_mass * p_mc2

        if enu0 >= 1.0:
            yield np.array([x0, y0, z0, enu0, azi0, pol0])
            num_good += 1


class EventSimulator(TrackerMixin, EventGeneratorMixin):
    def __init__(self, config, badpads=[]):
        super().__init__(config)
        self.untiltmat = pytpc.utilities.tilt_matrix(self.tilt)
        self.badpads = set(badpads)
        logger.info('%d pads will be dropped from the generated events', len(badpads))
        self.padmap = read_lookup_table(config['padmap_path'])  # maps (cobo, asad, aget, ch) -> pad
        self.reverse_padmap = {v: k for k, v in self.padmap.items()}  # maps pad -> (cobo, asad, aget, ch)

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


class EfficiencySimulator(object):
    def __init__(self, config, excluded_pads=[], lowgain_pads=[]):
        self.evtsim = EventSimulator(config, badpads=lowgain_pads)
        self.trigger = TriggerSimulator(config, excluded_pads=excluded_pads)
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
