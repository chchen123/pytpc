import numpy


class KalmanFilter:
    """ A class that defines a general extended Kalman filter.

    Attributes
    ----------

        sv_dim : The dimension of the state vector
        meas_dim : The dimension of the measurement vector
        q_mat : The process noise covariance matrix
        r_mat : The measurement noise covariance matrix
        xhat : The a posteriori state estimate
        p_mat : The estimated covariance matrix of xhat
        xhatminus : The a priori state estimate
        p_mat_minus : The estimated covariance matrix of xhatminus
        k_mat : The Kalman gain
        s_mat : The covariance matrix of the measurement innovation / residuals
        a_mat : The Jacobian matrix of the update function w.r.t. the state vector

        update : A callback function for estimating the next state vector
        jacobian : A callback function for estimating the Jacobian of the update function

    """

    def __init__(self, sv_dim, meas_dim, update, jacobian):
        """ Initializes the class.

        Arguments
        ---------

            sv_dim : The dimension of the state vector
            meas_dim : The dimension of the measurement vector
            update : A callback function for estimating the next state vector
            jacobian : A callback function for estimating the Jacobian of the update function
        """
        self.sv_dim = sv_dim
        self.meas_dim = meas_dim

        self._init_matrices(1)

        self.update = update
        self.jacobian = jacobian

    def _init_matrices(self, num_meas):
        """ Initializes the object's matrices.

        Arguments
        ---------
            num_meas : The number of samples in the data to be filtered
        """

        self.sv_sv_matsh = (num_meas, self.sv_dim, self.sv_dim)
        self.meas_meas_matsh = (num_meas, self.meas_dim, self.meas_dim)
        self.sv_meas_matsh = (num_meas, self.sv_dim, self.meas_dim)
        self.sv_shape = (num_meas, self.sv_dim)
        self.meas_shape = (num_meas, self.meas_dim)

        self.q_mat = numpy.eye(self.sv_dim)
        self.r_mat = numpy.eye(self.meas_dim)

        self.xhat = numpy.zeros(self.sv_shape)
        self.xhatminus = numpy.zeros(self.sv_shape)
        self.p_mat = numpy.zeros(self.sv_sv_matsh)
        self.p_mat_minus = numpy.zeros(self.sv_sv_matsh)
        self.k_mat = numpy.zeros(self.sv_meas_matsh)
        self.s_mat = numpy.zeros(self.meas_meas_matsh)
        self.i_mat = numpy.eye(self.sv_dim)
        self.a_mat = numpy.zeros(self.sv_sv_matsh)

    def apply(self, z):
        """ Apply the Kalman filter to the provided data set.

        Returns the state estimates for each point.
        """
        self._init_matrices(z.shape[0])

        self.xhat[0] = numpy.ones(self.sv_dim)
        self.p_mat[0] = numpy.eye(self.sv_dim) * 0.1

        for k in range(1, z.shape[0]):
            try:
                # time update step
                self.xhatminus[k] = self.update(self.xhat[k-1])  # + numpy.random.normal(0, 1e-2, 6)
                self.a_mat[k] = self.jacobian(self.xhatminus[k])

                a_mat_t = self.a_mat[k].T

                self.p_mat_minus[k] = numpy.dot(self.a_mat[k], numpy.dot(self.p_mat[k-1], a_mat_t)) + self.q_mat

                # measurement update step
                h_mat = numpy.eye(self.sv_dim, self.meas_dim)
                h_mat_t = h_mat.T

                self.s_mat[k] = numpy.dot(h_mat, numpy.dot(self.p_mat_minus[k], h_mat_t)) + self.r_mat
                self.k_mat[k] = numpy.dot(self.p_mat_minus[k], numpy.dot(h_mat_t, numpy.linalg.inv(self.s_mat[k])))
                self.xhat[k] = self.xhatminus[k] + numpy.dot(self.k_mat[k], (z[k] - self.xhatminus[k]))
                self.p_mat[k] = numpy.dot(self.i_mat - numpy.dot(self.k_mat[k], h_mat), self.p_mat_minus[k])
            except numpy.linalg.LinAlgError as err:
                print(err, k, self.s_mat[k])

        return self.xhat