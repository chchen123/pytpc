.. currentmodule:: pytpc

Interacting with Event Files
============================

The :mod:`evtdata` module is the main interface to event files. The functions and classes it contains can be used to open
an event file, read events out of it, and perform some basic processing on these events.

Basics
------

Opening Files
~~~~~~~~~~~~~

Most of the functions you'll need to read files are imported into the :mod:`pytpc` namespace by default, for
convenience. Therefore, to begin reading files, all you need to do is::

    import pytpc

A file can be opened by creating an instance of the :class:`evtdata.EventFile` class::

    ef = pytpc.EventFile('/path/to/event.evt')

The single argument to this constructor is a string representing the path to a valid event file. Any pre-merged event
file can be read. If the file's events have not had pedestals subtracted, though, you will need to do that manually.

The simplest way to read an event from a file is via subscripting::

    # Read the fifth event from the file
    evt = ef[5]

This is a very flexible method, as it also allows slicing, ::

    # Get events 100 through 106
    evts = ef[100:106]

and iterating through all of the events in a file::

    # Iterate over all events in the file, and do some_function
    for evt in ef:
        some_function(evt)

The subscripting and iterating methods shown above are the preferred (and easiest) way to work with the event files.
They are possible because the :class:`evtdata.EventFile` class automatically indexes the files when they are first
opened. These indices are stored next to the file in a separate file with the same name, but the extension
:file:`.lookup`.

Working with Events
~~~~~~~~~~~~~~~~~~~

Events are represented by the :class:`evtdata.Event` class. This is also imported into the :mod:`pytpc` namespace by
default, and it's what is returned when an event is read from a file.

An :class:`evtdata.Event` instance has a couple of important attributes. The :attr:`evt_id` attribute contains the
event ID, and the :attr:`timestamp` attribute contains the time stamp. The bulk of the data from the event is stored
in the :attr:`traces` attribute.

The traces are stored as a NumPy array with a complex data type. The different parts of each trace can be accessed using
field names:

``'cobo'``
    The CoBo number
``'asad'``
    The AsAd number
``'aget'``
    The AGET number
``'channel'``
    The channel of the AGET
``'pad'``
    The pad number, if it was assigned during data merging and pre-processing.
``'data'``
    The data itself. This is stored as an array of samples indexed by time bucket number.

The events also have some methods that manipulate the data into more useful forms. This includes producing a map of the
pad hits (:func:`evtdata.Event.hits`) and the 3-D track points in the TPC (:func:`evtdata.Event.xyzs`).
See the API reference for more information.

API Reference
-------------

pytpc.evtdata
~~~~~~~~~~~~~

This module contains the functions for working with event files.

.. currentmodule:: pytpc.evtdata

.. rubric:: Classes

.. autosummary::
    :toctree: generated/

    Event
    EventFile

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    calibrate_z
    uncalibrate_z
    load_pedestals
    load_padmap
