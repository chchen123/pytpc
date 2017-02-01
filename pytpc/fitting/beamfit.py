import numpy as np
from sklearn.linear_model import HuberRegressor


class BeamTrackLinearModel(object):
    def __init__(self, allowed_pads, zmin=0, zmax=1000):
        self.allowed_pads = allowed_pads
        self.zmin = zmin
        self.zmax = zmax

        self.xmodel = HuberRegressor()
        self.ymodel = HuberRegressor()

    def filter_data(self, data):
        data_filter = data.pad.isin(self.allowed_pads) & (data.z > self.zmin) & (data.z < self.zmax)
        return data[data_filter]

    def fit(self, data):
        fitdata = self.filter_data(data)

        zdata = fitdata.w.values.reshape(-1, 1)  # Required by scikit-learn's API
        xdata = fitdata.u
        ydata = fitdata.v

        self.xmodel.fit(zdata, xdata)
        self.ymodel.fit(zdata, ydata)

    @property
    def slopes(self):
        return np.array([self.xmodel.coef_[0], self.ymodel.coef_[0]])

    @property
    def intercepts(self):
        return np.array([self.xmodel.intercept_, self.ymodel.intercept_])

    def predict(self, zs):
        xs = self.xmodel.predict(zs)
        ys = self.ymodel.predict(zs)
        return np.column_stack((xs, ys))
