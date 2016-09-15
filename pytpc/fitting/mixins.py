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
    """A base class that discards all arguments sent to its initializer.

    The mixins in this file take arguments in their initializers, and they pass these on to the next class in the
    MRO using `super`. However, `object.__init__` does not take any arguments, so if one of these `super` calls
    reaches `object`, it will cause an error. Therefore the mixins inherit from this class instead.

    """
    def __init__(self, *args, **kwargs):
        pass


class TrackerMixin(MixinBase):
    """Provides a particle tracker to the inheriting class.

    This adds a particle tracker as `self.tracker` with associated attributes.

    """
    def __init__(self, config):
        self.gas = InterpolatedGas(config['gas_name'], config['gas_pressure'])
        self._efield = np.array(config['efield'])
        self._bfield = np.array(config['bfield'])
        self.mass_num = config['mass_num']
        self.charge_num = config['charge_num']
        self.beam_enu0 = config['beam_enu0']
        self.beam_mass = config['beam_mass']
        self.beam_charge = config['beam_charge']

        self.tracker = Tracker(self.mass_num,
                               self.charge_num,
                               self.beam_enu0,
                               self.beam_mass,
                               self.beam_charge,
                               self.gas,
                               self.efield,
                               self.bfield,
                               100)

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
    """Provides an event generator to the inheriting class.

    This adds an EventGenerator object as `self.evtgen`, with some other associated attributes.

    """
    def __init__(self, config):
        self._vd = np.array(config['vd'])
        self.mass_num = config['mass_num']
        self.pad_rot_angle = config['pad_rot_angle'] * degrees
        self.padrotmat = rot_matrix(self.pad_rot_angle)
        self.ioniz = config['ioniz']
        self.micromegas_gain = float(config['micromegas_gain'])
        self.electronics_gain = float(config['electronics_gain'])
        self.clock = config['clock']
        self.shape = float(config['shape'])
        self._tilt = config['tilt'] * degrees
        self.diff_sigma = config['diffusion_sigma']

        with h5py.File(config['lut_path'], 'r') as hf:
            lut = hf['LUT'][:]
        self.padplane = PadPlane(lut, -0.280, 0.0001, -0.280, 0.0001, self.pad_rot_angle)
        self.evtgen = EventGenerator(self.padplane, self.vd, self.clock, self.shape, self.mass_num,
                                     self.ioniz, self.micromegas_gain, self.electronics_gain,
                                     self.tilt, self.diff_sigma)

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
    """Function of a line for the ODR fitter."""
    return beta[0] * x + beta[1]


def line(x, a, b):
    """Function of a line."""
    return a * x + b


def find_linear_chi2(x, y, params):
    """Calculate chi2 for the linear prefit.

    Parameters
    ----------
    x, y : number or array-like
        The x and y coordinate values.
    params : iterable
        The parameters (a, b) to the `line` function, as returned from the fitter.

    """
    return np.sum((y - line(x, *params))**2) / (len(x) - 3)


def constrain_angle(ang):
    """Makes all values in `ang` be between 0 and 2π.

    Any values outside the [0, 2π) domain are reduced by a factor of 2π until they lie within the domain.

    Parameters
    ----------
    ang : array-like
        A set of angles to be constrained.

    Returns
    -------
    ndarray
        The angles from `ang` constrained to be between 0 and 2π.
    """
    x = ang % (2 * pi)
    x[x < 0] += 2 * pi
    return x


class LinearPrefitMixin(MixinBase):
    """Provides methods to perform a linear fit to the data to find an initial guess for the Monte Carlo parameters.

    Two methods are provided:
    1) `linear_prefit` -- This calculates r-phi and performs the linear fit.
    2) `guess_parameters` -- This uses the results of the linear fit to guess the starting point for the Monte Carlo.

    """
    def linear_prefit(self, xyz, cx, cy):
        """Performs the linear prefit.

        The linear fit is performed using the SciPy ODR library, which does orthogonal distance regression. This means
        that the minimized quantity is the orthogonal distance between the line and each data point, rather than the
        distance along one of the coordinate directions.

        Parameters
        ----------
        xyz : pandas.DataFrame
            The data to be fit. This must have columns (u, v, w) corresponding to the (x, y, z) positions in the
            *beam* reference frame. (I.e. after calibration and untilting.) This data is not copied, so some new
            columns will be added.
        cx, cy : number
            The position of the center of curvature, perhaps from a Hough space calculation.

        Returns
        -------
        res : dict
            The results of the fit. The dictionary has entries giving the fit parameters, chi^2, radius of curvature,
            B-rho, energy from radius of curvature, and center of curvature.

        """
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
        """Guess initial Monte Carlo parameters from the linear prefit results.

        This takes the dictionary from `linear_prefit` and creates a set of parameters to seed the Monte Carlo.

        Parameters
        ----------
        prefit_res : dict
            The output from `linear_prefit`.

        Returns
        -------
        ndarray
            The initial parameters (x0, y0, z0, enu0, azi0, pol0).

        """
        x0 = 0
        y0 = 0
        z0 = (prefit_res['lin_beam_int'] / 1000)
        enu0 = prefit_res['curv_en'] / self.mass_num
        azi0 = np.arctan2(-prefit_res['curv_ctr_y'], -prefit_res['curv_ctr_x']) - pi / 2
        pol0 = (pi - prefit_res['lin_scat_ang'])
        return np.array([x0, y0, z0, enu0, azi0, pol0], dtype='float64')


class PreprocessMixin(MixinBase):
    """Provides preprocessing (calibration) functions to its child class.
    """

    def __init__(self, config):
        self.micromegas_tb = config['micromegas_tb']
        super().__init__(config)

    def preprocess(self, raw_xyz, center=None, rotate_pads=False, last_tb=None):
        """Preprocesses data by calibrating it and un-tilting it.

        Parameters
        ----------
        raw_xyz : array-like
            The raw data. The array is assumed to be 2D with columns (x, y, z, a, pad) where `a` is the peak amplitude.
        center : list-like, optional
            The center of curvature. If provided, this will also be calibrated, and the calibrated value will be
            returned.
        rotate_pads : bool, optional
            If true, the points will be rotated about the z axis by the angle `self.pad_rot_angle`. This corresponds to
            the pad plane rotation. (Note that this may not be necessary depending on where `raw_xyz` was read from.)
        last_tb : int, optional
            The last time bucket to consider. If provided, all time buckets after this one will be zeroed. This
            can help with noise at the end of the event.

        Returns
        -------
        xyz : pandas.DataFrame
            The calibrated, un-tilted data. The DataFrame has columns `(u, v, w, a, pad)` (and others) giving
            the calibrated positions.
        center_uvw : ndarray, optional
            The calibrated center. This is only returned if the parameter `center` was not `None`.

        """
        raw_xyz = raw_xyz.copy()

        if last_tb is not None:
            raw_xyz = raw_xyz[raw_xyz[:, 2] < last_tb]  # get rid of noise at end

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
