Cleaning data
=============
..  currentmodule:: pytpc.cleaning

The :mod:`pytpc.cleaning` module contains the code used to remove noise from recorded data. This includes
the Hough transform code.

Main interface
--------------

In Python, data should be cleaned using either the :class:`HoughCleaner` or :class:`EventCleaner` class. These two
both provide Hough transform--based cleaning, but the :class:`EventCleaner` also has the ability to calibrate the
data before cleaning it. Another distinction is that :class:`HoughCleaner` manipulates calibrated coordinate data
(xyz triples), whereas the the :class:`EventCleaner` operates on :class:`~pytpc.evtdata.Event` objects, which
contain full traces.

..  rubric:: Cleaning classes
..  autosummary::
    :toctree: ./generated

    HoughCleaner
    EventCleaner


Low-level Python interface
--------------------------

Much of the cleaning code is computationally intensive and is implemented in C with Python wrappers. Consequently,
it is possible to call the Hough transform and nearest neighbor count functions directly, if desired. These
functions wrap multithreaded implementations that can be found in the file :file:`pytpc/cleaning/hough.c`.

..  warning::
    These functions are implemented in Cython, and are therefore a bit more fussy about their arguments than
    standard Python functions. Pay careful attention to their documentation.

..  rubric:: Low-level functions
..  autosummary::
    :toctree: ./generated

    hough_line
    hough_circle
    nearest_neighbor_count
