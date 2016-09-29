#!/usr/bin/env python

import numpy as np
from ..constants import degrees
from .mixins import TrackerMixin, EventGeneratorMixin, PreprocessMixin, LinearPrefitMixin
from .mcopt_wrapper import Annealer as McoptAnnealer
from . import BadEventError

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Annealer(PreprocessMixin, LinearPrefitMixin, TrackerMixin, EventGeneratorMixin):
    output_columns = [['x0', 'REAL'],
                      ['y0', 'REAL'],
                      ['z0', 'REAL'],
                      ['enu0', 'REAL'],
                      ['azi0', 'REAL'],
                      ['pol0', 'REAL'],
                      ['posChi2', 'REAL'],
                      ['enChi2', 'REAL'],
                      ['vertChi2', 'REAL'],
                      ['lin_scat_ang', 'REAL'],
                      ['lin_beam_int', 'REAL'],
                      ['lin_chi2', 'REAL'],
                      ['rad_curv', 'REAL'],
                      ['curv_en', 'REAL'],
                      ['curv_ctr_x', 'REAL'],
                      ['curv_ctr_y', 'REAL']]

    def __init__(self, config):
        self.beampads = np.fromfile(config['beampads_path'])

        sig = config['sigma']
        self.sigma = np.array([sig['x'],
                               sig['y'],
                               sig['z'],
                               sig['enu'],
                               sig['azi'] * degrees,
                               sig['pol'] * degrees])

        anneal_iters = int(config['anneal_num_iters'])
        cool_rate = float(config['cool_rate'])
        max_calls = int(config['max_calls_per_iter'])
        initial_temp = float(config['initial_temp'])

        self.annealer = McoptAnnealer(self.tracker, self.evtgen, initial_temp, cool_rate, anneal_iters, max_calls)

    def process_event(self, raw_xyz, cx, cy, exp_hits=None, remove_noise=True, return_details=False,
                      multi=False, preprocess_kwargs={}):
        xyz, (cu, cv) = self.preprocess(raw_xyz, center=[cx, cy], **preprocess_kwargs)

        if len(xyz) < 50:
            raise BadEventError("Not enough points")

        if exp_hits is None:
            exp_hits = np.zeros(10240)
            for a, p in xyz[['a', 'pad']].values:
                exp_hits[int(p)] = a

        xyz_sorted = xyz.sort_values(by='w', ascending=True)
        prefit_data = xyz_sorted.iloc[-len(xyz_sorted) // 4:].copy()

        prefit_res = self.linear_prefit(prefit_data, cu, cv)

        exp_pos = xyz_sorted[['u', 'v', 'w']].values.copy() / 1000

        ctr0 = self.guess_parameters(prefit_res)

        if multi:
            minfunc = self.annealer.multi_minimize
        else:
            minfunc = self.annealer.minimize

        annealres = minfunc(ctr0, self.sigma, exp_pos, exp_hits)

        ctr = annealres['ctrs'][-1]
        chis = annealres['chis'][-1]

        result = dict(zip(['x0', 'y0', 'z0', 'enu0', 'azi0', 'pol0'],
                          [float(v) for v in ctr]))
        result['posChi2'] = float(chis[0])
        result['enChi2'] = float(chis[1])
        result['vertChi2'] = float(chis[2])
        result.update(prefit_res)  # put the linear pre-fit results into result

        if return_details:
            return result, annealres
        else:
            return result
