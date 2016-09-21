"""A package for simulating, reading, and analyzing TPC data."""

import pytpc.evtdata
import pytpc.gases
import pytpc.grawdata
import pytpc.hdfdata
import pytpc.cleaning
import pytpc.fitting

from pytpc.evtdata import EventFile, Event
from pytpc.hdfdata import HDFDataFile
from pytpc.padplane import generate_pad_plane
