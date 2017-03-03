Fitting events
==============
..  currentmodule:: pytpc.fitting

The :mod:`fitting` module enables events to be fit with the Monte Carlo optimizer. This is one of the most
complex parts of the package since it calls into the ``mcopt`` external C++ library.

Main interface
--------------

The main class used to fit events is the :class:`MCFitter`. This class can preprocess and calibrate a set
of xyz data, fit it, and return the results as a dictionary.

..  rubric:: Fitter
..  autosummary::
    :toctree: ./generated

    MCFitter

The :class:`MCFitter` encompasses a lot of functionality, including particle tracking, event generation, fitting,
and preprocessing. Each of these parts can be used independently in your own classes by using the associated
mixins located in :mod:`pytpc.fitting.mixins`.

..  rubric:: Mixins
..  autosummary::
    :toctree: ./generated

    ~mixins.TrackerMixin
    ~mixins.EventGeneratorMixin
    ~mixins.LinearPrefitMixin
    ~mixins.PreprocessMixin
