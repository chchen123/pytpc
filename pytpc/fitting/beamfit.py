import numpy as np
from sklearn.linear_model import HuberRegressor


class BeamTrackLinearModel(object):
    def __init__(self, allowed_pads=np.arange(10240), zmin=0, zmax=1000, chi2norm=10.):
        self.allowed_pads = allowed_pads
        self.zmin = zmin
        self.zmax = zmax
        self.chi2norm = chi2norm

        self.xmodel = HuberRegressor()
        self.ymodel = HuberRegressor()

        self._num_pts = None
        self._chi2 = None

    def filter_data(self, data):
        data_filter = data.pad.isin(self.allowed_pads) & (data.z > self.zmin) & (data.z < self.zmax)
        return data[data_filter]

    def fit(self, filtered_data):
        fitdata = filtered_data[['u', 'v', 'w']].values

        xdata, ydata, zdata = fitdata.T
        zdata = zdata.reshape(-1, 1)

        self.xmodel.fit(zdata, xdata)
        self.ymodel.fit(zdata, ydata)

        self._num_pts = len(fitdata)
        self._chi2 = self.find_chi2(fitdata)

    @property
    def slopes(self):
        return np.array([self.xmodel.coef_[0], self.ymodel.coef_[0]])

    @property
    def intercepts(self):
        return np.array([self.xmodel.intercept_, self.ymodel.intercept_])

    def predict(self, zs):
        zs = np.asanyarray(zs)
        if zs.ndim == 1:
            zs = zs.reshape(-1, 1)
        xs = self.xmodel.predict(zs)
        ys = self.ymodel.predict(zs)
        return np.column_stack((xs, ys))

    def find_chi2(self, data, sigma=10.):
        predicted = self.predict(data[:, 2])
        return np.sum(np.square(predicted - data[:, :2])) / (len(data) * sigma**2)

    @property
    def chi2(self):
        if self._chi2 is not None:
            return self._chi2
        else:
            raise RuntimeError('Undefined. Fit data first.')

    @property
    def num_pts(self):
        if self._num_pts is not None:
            return self._num_pts
        else:
            raise RuntimeError('Undefined. Fit data first')
