"""A package for simulating, reading, and analyzing TPC data."""

from pytpc.constants import *
import pytpc.tracking
import pytpc.simulation
import pytpc.evtdata
import pytpc.tpcplot

from pytpc.tracking import find_tracks, Tracker
from pytpc.simulation import Particle, Gas, track
from pytpc.evtdata import EventFile, Event
from pytpc.tpcplot import chamber_plot, pad_plot
from pytpc.padplane import generate_pad_plane