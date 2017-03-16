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

Mixins
------

The :class:`MCFitter` encompasses a lot of functionality, including particle tracking, event generation, fitting,
and preprocessing. Each of these parts can be used independently in your own classes by using the associated
`mixins <https://en.wikipedia.org/wiki/Mixin>`_ located in :mod:`pytpc.fitting.mixins`. The mixins each contain a
fragment of the functionality of the :class:`MCFitter` class, and each of them depends on the analysis config
file for its parameters. For example, to include particle tracking and event generation capabilities in a custom class,
inherit from :class:`TrackerMixin` and :class:`EventGeneratorMixin`:

..  code-block:: python

    class ExampleClass(TrackerMixin, EventGeneratorMixin):
        def __init__(self, config):
            super().__init__(config)  # Initializes the Tracker and EventGenerator
            # Custom initialization for ExampleClass goes here

The resulting class will have a :class:`Tracker` member named ``tracker`` and an :class:`EventGenerator` member
named ``evtgen``, along with some other members containing the simulation parameters from the config file.

..  rubric:: Mixins
..  autosummary::
    :toctree: ./generated

    ~mixins.TrackerMixin
    ~mixins.EventGeneratorMixin
    ~mixins.LinearPrefitMixin
    ~mixins.PreprocessMixin
