"""montecarlo.py

This module contains the main Python class for fitting data with the Monte Carlo algorithm.

"""

import numpy as np
from ..constants import degrees
from .mixins import TrackerMixin, EventGeneratorMixin, PreprocessMixin, LinearPrefitMixin
from . import BadEventError
from .mcopt_wrapper import Minimizer

import logging
logger = logging.getLogger(__name__)


class MCFitter(PreprocessMixin, LinearPrefitMixin, TrackerMixin, EventGeneratorMixin):
    """Performs Monte Carlo fits on events using the C++ fitting code.

    Parameters
    ----------
    config : dict
        The analysis config dictionary.

    """
    def __init__(self, config):
        super().__init__(config)

        #: A set of pads that can be removed during preprocessing.
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
        """Fit the given dataset using the Monte Carlo algorithm.

        The details of the fit are governed by the Monte Carlo parameters ``num_iters``, ``num_pts``, and
        ``red_factor``, which can be adjusted by setting the corresponding properties on this class.

        Parameters
        ----------
        xyz : pd.DataFrame
            The data in the beam coordinate system. Must have columns 'u', 'v', and 'w' for the position, in mm,
            'a' for the amplitude, and 'pad' for the pad number.
        cu, cv : float
            The center of curvature of the track, in mm.
        exp_hits : np.ndarray, optional
            The hit pattern of the event. If None, a hit pattern will be calculated from ``xyz``.
        return_details : bool, optional
            If true, more data will be returned in addition to the result. See below for details.

        Returns
        -------
        result : dict
            The fit result. Has keys for the 6 track parameters, the 3 objective function components,
            and some information about the prefit results.
        min_chis : np.ndarray
            The minimum total chi2 value for each iteration. Only returned if ``return_details == True``.
        all_params : np.ndarray
            The parameters from all generated tracks. There will be `num_iters * num_pts` rows.
            Only returned if ``return_details == True``.
        good_param_idx : np.ndarray
            The row numbers in ``all_params`` corresponding to the best points from each iteration, i.e. the ones whose
            chi2 values are in ``min_chis``. Only returned if ``return_details == True``.

        """
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
