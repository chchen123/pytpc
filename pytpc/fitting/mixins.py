from . import Tracker, EventGenerator, PadPlane
import numpy as np
import pandas as pd
from scipy import odr
from pytpc.evtdata import calibrate
from pytpc.gases import InterpolatedGas
from pytpc.constants import degrees, pi, e_chg, p_kg
from pytpc.utilities import rot_matrix, tilt_matrix
import h5py


class MixinBase(object):
    def __init__(self, *args, **kwargs):
        pass


class TrackerMixin(MixinBase):
    def __init__(self, config):
        self.gas = InterpolatedGas(config['gas_name'], config['gas_pressure'])
        self._efield = np.array(config['efield'])
        self._bfield = np.array(config['bfield'])
        self.mass_num = config['mass_num']
        self.charge_num = config['charge_num']
        self.beam_enu0 = config['beam_enu0']
        self.beam_mass = config['beam_mass']
        self.beam_charge = config['beam_charge']

        self.tracker = Tracker(mass_num=self.mass_num,
                               charge_num=self.charge_num,
                               beam_enu0=self.beam_enu0,
                               beam_mass=self.beam_mass,
                               beam_charge=self.beam_charge,
                               gas=self.gas,
                               efield=self.efield,
                               bfield=self.bfield,
                               max_en=100)

        super().__init__(config)

    @property
    def efield(self):
        return self._efield

    @efield.setter
    def efield(self, value):
        value = np.asarray(value, dtype='float64')
        self.tracker.efield = value
        self._efield = value

    @property
    def bfield(self):
        return self._bfield

    @bfield.setter
    def bfield(self, value):
        value = np.asarray(value, dtype='float64')
        self.tracker.bfield = value
        self._bfield = value

    @property
    def bfield_mag(self):
        return np.linalg.norm(self.bfield)


class EventGeneratorMixin(MixinBase):
    def __init__(self, config):
        self._vd = np.array(config['vd'])
        self.mass_num = config['mass_num']
        self.pad_rot_angle = config['pad_rot_angle'] * degrees
        self.padrotmat = rot_matrix(self.pad_rot_angle)
        self.ioniz = config['ioniz']
        self.gain = config['micromegas_gain']
        self.clock = config['clock']
        self.shape = float(config['shape'])
        self._tilt = config['tilt'] * degrees
        self.diff_sigma = config['diffusion_sigma']

        with h5py.File(config['lut_path'], 'r') as hf:
            lut = hf['LUT'][:]
        self.padplane = PadPlane(lut, -0.280, 0.0001, -0.280, 0.0001, self.pad_rot_angle)
        self.evtgen = EventGenerator(self.padplane, self.vd, self.clock, self.shape, self.mass_num,
                                     self.ioniz, self.gain, self.tilt, self.diff_sigma)

        super().__init__(config)

    @property
    def vd(self):
        return self._vd

    @vd.setter
    def vd(self, value):
        value = np.asarray(value, dtype='float64')
        self.evtgen.vd = value
        self._vd = value

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        self.evtgen.tilt = value
        self._tilt = value


def odrline(beta, x):
    return beta[0] * x + beta[1]


def line(x, a, b):
    return a * x + b


def find_linear_chi2(x, y, params):
    return np.sum((y - line(x, *params))**2) / (len(x) - 3)


def constrain_angle(ang):
    x = ang % (2 * pi)
    x[x < 0] += 2 * pi
    return x


class LinearPrefitMixin(MixinBase):
    def linear_prefit(self, xyz, cx, cy):
        # Recenter the data on the center of curvature and find cylindrical coordinates in this system
        xyz['cx'] = xyz.u - cx
        xyz['cy'] = xyz.v - cy
        xyz['cr'] = np.hypot(xyz.cx, xyz.cy)
        xyz['cth'] = np.unwrap(constrain_angle(np.arctan2(-cy, -cx) - np.arctan2(xyz.cy, xyz.cx)), discont=1.8 * pi)
        if np.abs(xyz['cth'].min() - 2 * pi) < 5 * degrees:
            xyz['cth'] -= 2 * pi

        # Fit the arc length vs z data with a line
        ydata = xyz.cr * xyz.cth
        odrdata = odr.Data(ydata, xyz.w, wd=xyz.w**2, we=xyz.w**2)
        odrmodel = odr.Model(odrline)
        odrfitter = odr.ODR(odrdata, odrmodel, beta0=[-0.5, 500])
        odrresults = odrfitter.run()
        if odrresults.info >= 5:
            raise RuntimeError("ODR prefit failed: {}".format(odrresults.stopreason))
        lparams = odrresults.beta
        # lcov = odrresults.cov_beta
        ang = pi / 2 + np.arctan(lparams[0])
        beam_intercept = lparams[1]

        chi2 = find_linear_chi2(ydata, xyz.w, lparams)

        rad_curv = xyz.cr.mean()
        brho = rad_curv / np.sin(ang) * self.bfield_mag / 1e3
        curv_en = (brho**2 * self.charge_num**2 * e_chg**2 / (2 * self.mass_num * p_kg)) / e_chg * 1e-6

        res = {'lin_scat_ang': float(ang),
               'lin_beam_int': float(beam_intercept),
               'lin_chi2': float(chi2),
               'rad_curv': float(rad_curv),
               'brho': float(brho),
               'curv_en': float(curv_en),
               'curv_ctr_x': float(cx),
               'curv_ctr_y': float(cy)}

        return res

    def guess_parameters(self, prefit_res):
        x0 = 0
        y0 = 0
        z0 = (prefit_res['lin_beam_int'] / 1000)
        enu0 = prefit_res['curv_en'] / self.mass_num
        azi0 = np.arctan2(-prefit_res['curv_ctr_y'], -prefit_res['curv_ctr_x']) - pi / 2
        pol0 = (pi - prefit_res['lin_scat_ang'])
        return np.array([x0, y0, z0, enu0, azi0, pol0], dtype='float64')


class PreprocessMixin(MixinBase):
    
    def __init__(self, config):
        self.micromegas_tb = config['micromegas_tb']
        super().__init__(config)

    def preprocess(self, raw_xyz, center=None, rotate_pads=False, last_tb=None):
        raw_xyz = raw_xyz.copy()

        if last_tb is not None:
            raw_xyz = raw_xyz[raw_xyz[:, 2] < 505]  # get rid of noise at end

        if rotate_pads:
            raw_xyz[:, 0], raw_xyz[:, 1] = self.padrotmat @ raw_xyz[:, :2].T
            if center is not None:
                center = self.padrotmat @ center

        # Correct for shift in z due to trigger delay
        raw_xyz[:, 2] -= self.micromegas_tb

        # Correct for Lorentz angle and find z dimension
        # xyz = pd.DataFrame(pytpc.evtdata.calibrate(raw_xyz, self.vd, self.clock), columns=('x', 'y', 'z', 'a', 'pad'))
        xyz = pd.DataFrame(calibrate(raw_xyz[np.where(~np.in1d(raw_xyz[:, -1], self.beampads))],
                                     self.vd, self.clock), columns=('x', 'y', 'z', 'a', 'pad'))

        # Find the untilted coordinates
        tmat = tilt_matrix(-self.tilt)
        xyz['u'], xyz['v'], xyz['w'] = np.inner(tmat, xyz[['x', 'y', 'z']])
        xyz['v'] += np.tan(self.tilt) * 1000.  # Corrects for rotating around umegas instead of cathode

        if center is not None:
            center_uvw = calibrate(np.array([[center[0], center[1], 0]]), self.vd, self.clock).ravel()
            center_uvw = tmat @ center_uvw
            center_uvw[1] += np.tan(self.tilt) * 1000.  # Corrects for rotating around umegas instead of cathode

            return xyz, center_uvw[:2]

        else:
            return xyz
