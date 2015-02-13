"""
ukf
===

Implements an Unscented Kalman Filter. This code is based on the corresponding
class from the filterpy library (http://github.com/rlabbe/filterpy).

"""

from __future__ import division, print_function
import numpy as np
from numpy.linalg import cholesky, inv


class UnscentedKalmanFilter(object):

    def __init__(self, dim_x, dim_z, fx, hx):
        self._dim_x = dim_x  #: The state vector dimension
        self._dim_z = dim_z  #: The measurement vector dimension
        self.fx = fx         #: The prediction function
        self.hx = hx         #: The measurement update function

        self.Q = np.eye(dim_x)  #: The process noise matrix
        self.R = np.eye(dim_z)  #: The measurement noise matrix

        self.x = np.zeros(dim_x)  #: The estimated state vector
        self.P = np.eye(dim_x)    #: The estimated covariance matrix

        self.kappa = 0  #: A fine-tuning parameter
        self.W = self.find_weights(dim_x, self.kappa)

        self.sigmas_f = np.zeros((2*dim_x + 1, dim_x))

    def predict(self):

        # Calculate sigma points
        sigmas = self.find_sigma_points(self.x, self.P, self.kappa)

        # Pass sigma points through prediction function
        self.sigmas_f = self.fx(sigmas)

        # Find the weighted mean and covar matrix
        self.x, self.P = self.unscented_transform(self.sigmas_f, self.W, self.Q)

    def update(self, z):

        # Run predicted points through measurement function
        sigmas_h = self.hx(self.sigmas_f)

        # This is the predicted observation and its covar
        zp, Pz = self.unscented_transform(sigmas_h, self.W, self.R)

        # This finds the cross-correlation matrix between x and z
        yh = self.sigmas_f - self.x[np.newaxis, :]
        yz = sigmas_h - zp[np.newaxis, :]
        Pxz = yh.T.dot(np.diag(self.W)).dot(yz)

        # The Kalman gain and innovation
        K = np.dot(Pxz, inv(Pz))
        y = z - zp

        # The updated results
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(Pz, K.T))

    @staticmethod
    def find_sigma_points(x, P, k):

        x = np.asanyarray(x)
        P = np.asanyarray(P)

        n = np.size(x)

        sigmas = np.zeros((2*n+1, n))

        u = cholesky((n+k) * P).T

        sigmas[0] = x
        sigmas[1:n+1] = x + u
        sigmas[n+1:2*n+2] = x - u

        return sigmas

    @staticmethod
    def find_weights(n, kappa):

        w = np.full(2*n+1, 0.5 / (n+kappa))
        w[0] = kappa / (n+kappa)
        return w

    @staticmethod
    def unscented_transform(sigmas, weights, noise_covar):

        # Calculate weighted mean
        x = np.dot(weights, sigmas)  # this is sum of W_i * x_i

        # Calculate the covariance
        y = sigmas - x[np.newaxis, :]
        P = y.T.dot(np.diag(weights)).dot(y)
        if noise_covar is not None:
            P += noise_covar

        return x, P