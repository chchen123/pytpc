Coordinate systems and angles
=============================

Between the various fields, perspectives, and the tilt angle, the coordinate system used in the analysis can
quickly become quite confusing. It's important to define it well to begin with. This code assumes the coordinate
system described on this page.

General coordinate system
-------------------------

This code uses Cartesian coordinates. The z direction is along the drift axis of the TPC, pointing upstream from
the Micromegas. This implies that electrons will drift in a *negative z* direction as time passes.

The x and y directions are defined by the Micromegas. Take the x direction to be horizontal on the pad plane and y to
be vertical.

..  caution::
    It's important to remember that the Micromegas is rotated when it is installed in the TPC. Therefore, the
    x and y dimensions of the generated pad plane points (see :func:`pytpc.padplane.generate_pad_plane`) are not
    the same as the horizontal and vertical directions in the lab unless a rotation is applied.

    The connector side of the Micromegas has 72-degree rotational symmetry. When the Micromegas is installed in the
    TPC, it is rotated through 1.5 of these symmetric segments, so the rotation angle for the transformation is
    --108 degrees.

Using this convention, the beam, fields, and drift have the following alignment:

.. cssclass:: table-striped

============  =============
  Quantity      Direction
============  =============
B field       Along +z
E field       Along +z
Beam          Along -z
Drift         Along -z
============  =============

Tilting the detector
--------------------

Tilting the detector presents a special problem since it makes the electric and magnetic fields non-parallel. The
easiest way to deal with this is to keep the detector's coordinate system as above, and *tilt the magnetic field vector*
instead. This is illustrated in the figure below.

.. figure:: images/CoordinateSystems.svg
    :width: 400 px
    :align: center
    :figwidth: 600 px

    The coordinate system when the detector is tilted. Here, :math:`\tau` is the tilt angle. From the perspective
    of the detector's coordinate system, the magnetic field vector is tilted upward by :math:`\tau`.

Using this system, the vectors are along these axes:

.. cssclass:: table-striped

============  ==================================
  Quantity      Direction
============  ==================================
B field       Components in +y and +z directions
E field       Along +z
Beam          Components in -y and -z directions
Drift         Components in all three directions
============  ==================================

