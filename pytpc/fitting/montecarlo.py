#!/usr/bin/env python

import numpy as np
from ..constants import degrees
from .mixins import TrackerMixin, EventGeneratorMixin, PreprocessMixin, LinearPrefitMixin
from . import BadEventError
from .mcopt_wrapper import Minimizer

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class MCFitter(PreprocessMixin, LinearPrefitMixin, TrackerMixin, EventGeneratorMixin):
    def __init__(self, config):
        super().__init__(config)

        self.beampads = np.fromfile(config['beampads_path'], sep=',', dtype='int')

        sig = config['sigma']
        self.sigma = np.array([sig['x'],
                               sig['y'],
                               sig['z'],
                               sig['enu'],
                               sig['azi'] * degrees,
                               sig['pol'] * degrees])
        num_iters = config['num_iters']
        num_pts = config['num_pts']
        red_factor = config['red_factor']

        self.minimizer = Minimizer(self.tracker, self.evtgen, num_iters, num_pts, red_factor)

    def process_event(self, xyz, cu, cv, exp_hits=None, return_details=False):
        if exp_hits is None:
            exp_hits = np.zeros(10240)
            for a, p in xyz[['a', 'pad']].values:
                exp_hits[int(p)] = a

        xyz_sorted = xyz.sort_values(by='w', ascending=True)
        prefit_data = xyz_sorted.iloc[-len(xyz_sorted) // 4:].copy()

        prefit_res = self.linear_prefit(prefit_data, cu, cv)

        exp_pos = xyz_sorted[['u', 'v', 'w']].values.copy() / 1000

        ctr0 = self.guess_parameters(prefit_res)

        minres = self.minimizer.minimize(
            ctr0,
            self.sigma,
            exp_pos,
            exp_hits,
            details=return_details
        )

        if return_details:
            ctr, min_chis, all_params, good_param_idx = minres
            posChi = min_chis[-1, 0]
            enChi = min_chis[-1, 1]
            vertChi = min_chis[-1, 2]
        else:
            ctr, posChi, enChi, vertChi = minres

        result = dict(zip(['x0', 'y0', 'z0', 'enu0', 'azi0', 'pol0'],
                          [float(v) for v in ctr]))
        result['posChi2'] = float(posChi)
        result['enChi2'] = float(enChi)
        result['vertChi2'] = float(vertChi)

        result.update(prefit_res)  # put the linear pre-fit results into result

        if return_details:
            return result, min_chis, all_params, good_param_idx
        else:
            return result

    @property
    def num_iters(self):
        return self.minimizer.num_iters

    @num_iters.setter
    def num_iters(self, value):
        self.minimizer.num_iters = value

    @property
    def num_pts(self):
        return self.minimizer.num_pts

    @num_pts.setter
    def num_pts(self, value):
        self.minimizer.num_pts = value

    @property
    def red_factor(self):
        return self.minimizer.red_factor

    @red_factor.setter
    def red_factor(self, value):
        self.minimizer.red_factor = value
