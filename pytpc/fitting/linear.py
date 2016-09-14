import numpy as np
from scipy import odr
from ..constants import degrees, pi, e_chg, p_kg


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


class LinearPrefitMixin(object):
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
