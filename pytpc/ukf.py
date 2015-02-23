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
    """Represents an unscented Kalman filter, as defined by [1]_.

    Parameters
    ----------
    dim_x : int
        The dimension of the state vector
    dim_z : int
        The dimension of the measurement vector
    fx : function(sv, dt)
        The prediction function. This should take the current state vector as an array-like object and the timestep
        to the next measured point as a float. It should return the next state vector.
    hx : function(sv)
        The measurement function. This should take a state vector as an array-like object and return a measured point
        with the dimension dim_x.
    dtx : function(sv, dpos)
        A function that takes the current state vector and the scalar distance to the next data point and returns
        the timestep required to get to the next data point.

    Attributes
    ----------
    Q
    R
    P
    x

    References
    ----------
    ..  [1] Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems.
            In I. Kadar (Ed.), SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI.
            doi:10.1117/12.280797
    """

    def __init__(self, dim_x, dim_z, fx, hx, dtx):
        self._dim_x = dim_x  #: The state vector dimension
        self._dim_z = dim_z  #: The measurement vector dimension
        self.fx = fx         #: The prediction function
        self.hx = hx         #: The measurement update function
        self.dtx = dtx

        self.Q = np.eye(dim_x)
        """The process noise matrix."""

        self.R = np.eye(dim_z)  #: The measurement noise matrix

        self.x = np.zeros(dim_x)  #: The estimated state vector
        self.P = np.eye(dim_x)    #: The estimated covariance matrix

        self.kappa = 3 - dim_x  #: A fine-tuning parameter
        self.W = self.find_weights(dim_x, self.kappa)

        self.sigmas_f = np.zeros((2*dim_x + 1, dim_x))

    def predict(self, dt):
        """Predict the next state vector, based on the current state vector.

        This function returns None, but it stores its results in self.x and self.P.

        Parameters
        ----------
        dt : float
            The time step to the next measured point.
        """
        # Calculate sigma points
        sigmas = self.find_sigma_points(self.x, self.P, self.kappa)

        # Pass sigma points through prediction function
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.fx(s, dt)

        # Find the weighted mean and covar matrix
        self.x, self.P = self.unscented_transform(self.sigmas_f, self.W, self.Q)

    def update(self, z):
        """Update the state vector to include the information from the next measured point.

        This function reads the predicted state from self.x and self.P and updates that information based on the
        provided measurement point.

        Parameters
        ----------
        z : array-like
            The measured point. It should have dimension equal to (dim_z).
        """

        # Run predicted points through measurement function
        sigmas_h = np.zeros((self.sigmas_f.shape[0], self._dim_z))
        for i, s in enumerate(self.sigmas_f):
            sigmas_h[i] = self.hx(s)

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

    def batch_filter(self, zs):
        """Apply the filter to a set of measured points.

        This is the typical method of using the filter class. The filter is applied by iterating through the points
        and running the predict and update functions on each one.

        Parameters
        ----------
        zs : array-like
            The measured data points to be filtered. The dimension should be (n, dim_z) where n is the number of
            measured points.

        Returns
        -------
        means : ndarray
            The mean points from the filter (i.e. the fitted result of the filter)
        covars : ndarray
            The covariance matrices from each filtered point. These can be used to assess the error in the fit, and
            to judge how well the filter converged.
        times : ndarray
            The time at each fitted point, as determined by the calculated time step between each data point. These
            can be used as the x-axis in a plot.
        """

        zs = np.asanyarray(zs)

        n = np.size(zs, 0)

        means = np.zeros((n, self._dim_x))
        covars = np.zeros((n, self._dim_x, self._dim_x))
        times = np.zeros(n)
        current_time = 0.

        delta_zs = np.zeros((n, self._dim_z))
        delta_zs[0, :] = zs[0] - self.hx(self.x)
        delta_zs[1:, :] = np.diff(zs, axis=0)

        dpos = np.linalg.norm(zs[-1] - zs[0]) / n  # average is total length / number of pts

        for i in range(n):
            dt = self.dtx(self.x, dpos)
            self.predict(dt)
            self.update(zs[i])
            means[i, :] = self.x
            covars[i, :, :] = self.P
            current_time += dt
            times[i] = current_time

        return means, covars, times

    @staticmethod
    def find_sigma_points(x, P, k):
        r"""Finds the sigma points for the unscented transform.

        These are a set of points scattered around the state space. They are used to estimate the mean and covariance
        after the predict and update functions are applied.

        The sigma points are defined as follows: [1]_

        ..  math::
            \mathbf{\Sigma}_0 &= \overline{\mathbf{x}} \\
            \mathbf{\Sigma}_i &= \overline{\mathbf{x}} + \left( \sqrt{(n + \kappa) \mathbf{P}_{xx}} \right)_i \\
            \mathbf{\Sigma}_{i+n} &= \overline{\mathbf{x}} - \left( \sqrt{(n + \kappa) \mathbf{P}_{xx}} \right)_i

        Here, :math:`\hat{\mathbf{x}}` is the mean at the given point, :math:`\mathbf{P_{xx}}` is the covariance matrix
        of the variable :math:`\mathbf{x}`, :math:`n` is the dimension of the vector :math:`\mathbf{x}`, and the square
        root is any matrix square root function. Here, I use the Cholesky decomposition.

        Parameters
        ----------
        x : array-like
            The state vector, or the means before the transform
        P : array-like
            The covariance matrix corresponding to x
        k : float
            A tuning parameter. See [1]_ for details.

        Returns
        -------
        sigmas : ndarray
            The sigma points, as defined above

        References
        ----------
        ..  [1] Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems.
                In I. Kadar (Ed.), SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI.
                doi:10.1117/12.280797
        """

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
        r"""Find the weights for the sigma points.

        The weights are defined as follows:

        ..  math::
            W_0 &= \frac{\kappa}{n+\kappa} \\
            W_i &= \frac{1}{2(n+\kappa)} \\
            W_{i+n} &= \frac{1}{2(n+\kappa)}

        The indices in the math above correspond to the indices of the sigma points from the :func:`find_sigma_points`
        function.

        Parameters
        ----------
        n : int
            The dimension of the state vector, or the dimension of the vector :math:`\mathbf{x}` provided to
            :func:`find_sigma_points`
        kappa : int
            A fine-tuning parameter

        Returns
        -------
        w : ndarray
            The weights
        """

        w = np.full(2*n+1, 0.5 / (n+kappa))
        w[0] = kappa / (n+kappa)
        return w

    @staticmethod
    def unscented_transform(sigmas, weights, noise_covar):
        r"""Performs the unscented transformation.

        This function uses the provided sigma points, weights, and noise covariance matrix to find the weighted averages
        for the unscented transformation.

        The weighted mean is defined as: [1]_

        ..  math::
            \overline{\mathbf{y}} = \sum_{i} W_i \mathbf{\Sigma}_i

        and the covariance of this is

        ..  math::
            \mathbf{P}_{yy} = \sum_i W_i \{\mathbf{\Sigma}_i - \overline{\mathbf{y}}\}
                \{\mathbf{\Sigma}_i - \overline{\mathbf{y}}\}^T

        Parameters
        ----------
        sigmas : array-like
            The sigma points
        weights : array-like
            The weights for each sigma point
        noise_covar : array-like
            An external noise covariance matrix to be included in the weighted average.

        Returns
        -------
        x : ndarray
            The transformed mean
        P : ndarray
            The transformed covariance matrix

        See Also
        --------
        find_sigma_points : Can be used to find sigma points
        find_weights : Can be used to find the weights

        References
        ----------
        ..  [1] Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems.
                In I. Kadar (Ed.), SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI.
                doi:10.1117/12.280797
        """

        # Calculate weighted mean
        x = np.dot(weights, sigmas)  # this is sum of W_i * x_i

        # Calculate the covariance
        y = sigmas - x[np.newaxis, :]
        P = y.T.dot(np.diag(weights)).dot(y)
        if noise_covar is not None:
            P += noise_covar

        return x, P