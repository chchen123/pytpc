..  currentmodule:: pytpc

Unscented Kalman Filter
=======================

This module implements the Unscented Kalman Filter. The implementation itself is a slightly modified version of the
:mod:`filterpy` package. [3]_

Basic Idea
----------

The Kalman filter is described in great detail in other places, but the basic idea of its operation is a
two-step cycle: [1]_

1. Using the current knowledge of the state of a system, **predict** its next state
2. Use a measured data point to **update** this prediction to reflect reality

This *predict* and *update* cycle is run for each data point. The result is a good estimation of the actual state
of the system at each time step, with an associated covariance matrix.

The *unscented Kalman filter* is a particular subtype of the general Kalman filter that deals with nonlinear
systems [2]_. Its benefit is that it doesn't require the system to be explicitly linearized, and it instead estimates
the properties of the system using the *unscented transform*.

.. _ukf_vocab:

Vocabulary
----------

The Kalman filter literature (along with this code) uses a relatively standard set of variables to represent quantities.
Some of the relevant ones are shown below:

x
    The state vector. It has dimension *n*.
z
    The measurement vector. It has dimension *m*.
P
    The covariance of the state vector, dimension *n* by *n*.
Q
    The covariance matrix of the noise in the process (physical) model, dimension *n* by *n*.
R
    The covariance matrix of the noise in the measurements, dimension *m* by *m*.
f(x)
    A function that predicts the next state vector
h(z)
    A function that converts a state vector into a measurement vector

Usage
-----

..  note::
    It's probably better to use the filter through the :class:`Tracker` wrapper class in :mod:`pytpc.tracking` instead
    of using it directly. Still, the basic ideas outlined below apply.

The filter can be instantiated as follows::

    kf = pytpc.ukf.UnscentedKalmanFilter(dim_x, dim_z, fx, hx, dtx)

Here, ``dim_x`` is the dimension of the state vector, ``dim_z`` is the dimension of the measurement vector, ``fx`` is
a function that predicts the next state, ``hx`` is a function that turns a state vector into a measurement, and ``dtx``
is a function that returns the time step to be used in the prediction step.

After instantiating the class, we need to set the **Q** and **R** matrices and provide an initial guess for the first
**x** vector and **P** matrix. Do this using NumPy arrays of the appropriate dimensions (see :ref:`ukf_vocab` above).

Finally, we can process data. Generally, we want to process a list of data that was taken previously. This can be done
with :func:`batch_filter`::

    data = get_data_from_somewhere()  # perhaps this came from an Event
    means, covars, times = kf.batch_filter(data)

The outputs are the fitted data points (or "means"), covariance matrices at each point, and the time at each point. The
last part is mainly convenient for plotting.

API Reference
-------------

pytpc.ukf
~~~~~~~~~

..  currentmodule:: pytpc.ukf

An implementation of the unscented Kalman filter.

..  rubric:: Classes

..  autosummary::
    :toctree: generated/

    UnscentedKalmanFilter

References
----------

..  [1] https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
..  [2] Julier, S. J., & Uhlmann, J. K. (1997). A New Extension of the Kalman Filter to Nonlinear Systems.
    In I. Kadar (Ed.), SPIE 3068, Signal Processing, Sensor Fusion, and Target Recognition VI. doi:10.1117/12.280797
..  [3] http://github.com/rlabbe/filterpy