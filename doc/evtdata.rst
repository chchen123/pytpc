.. currentmodule:: pytpc

Interacting with Event Files -- `evtdata`
=========================================

The `evtdata` module is the main interface to event files. The functions and classes it contains can be used to open
an event file, read events out of it, and perform some basic processing on these events.

The basic method to open an read a file is as follows::

    # The Event and EventFile classes are included in the pytpc package
    # namespace for convenience.
    import pytpc

    # Open an event file
    ef = pytpc.EventFile('/path/to/event.evt')

    # Read the fifth event from the file
    evt = ef[5]

    # Iterate over all events in the file, and do some_function
    for evt in ef:
        some_function(evt)

    # Get events 100 through 106
    evts = ef[100:106]

The subscripting and iterating methods shown above are the preferred (and easiest) way to work with the event files.

.. rubric:: Classes

.. autosummary::
    :toctree: generated/

    evtdata.Event
    evtdata.EventFile

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    evtdata.calibrate_z
    evtdata.uncalibrate_z
    evtdata.load_pedestals
    evtdata.load_padmap
