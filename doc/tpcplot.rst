..  currentmodule:: pytpc

Plotting Data -- `tpcplot`
==========================

This module contains functions for displaying events from the AT-TPC. The main two are `chamber_plot`, which produces
a three-dimensional view of the event's tracks, and `pad_plot`, which shows the hit pattern on the Micromegas pad plane.

These plotting functions are best used in conjunction with the methods of the `Event` class, though they can also work
independently.

Examples
--------

A file and an event can be opened and read in the usual way::

    import pytpc  # All the functions we need are in the package scope
    ef = pytpc.EventFile('/path/to/data.evt')  # The event file
    evt = ef[0]  # The event

Then, if we want to see the pad plane, we could do this::

    pad_fig = pytpc.pad_plot(evt.hits())

For the three-dimensional view of the tracks, run::

    chamber_fig = pytpc.chamber_plot(evt.xyzs())

More information about the `hits` and `xyzs` functions can be found in the documentation of the :class:`Event` class.

.. rubric:: Functions

..  autosummary::
    :toctree: generated/

    tpcplot.pad_plot
    tpcplot.chamber_plot
    tpcplot.show_pad_plane