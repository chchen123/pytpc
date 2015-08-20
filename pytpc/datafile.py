"""
datafile
========

This module provides a base class for interacting with data files.

"""

from __future__ import division, print_function
import os.path


class DataFile(object):

    def __init__(self, filename=None, open_mode='r'):

        self.lookup = []
        """A lookup table for the events in the file. This is simply an array of file offsets."""

        self.current_event = 0  #: The index of the current event

        if filename is not None:
            self.open(filename, open_mode)
            if 'r' in open_mode:
                ltfilename = filename + '.lookup'
                if os.path.exists(ltfilename):
                    self.load_lookup_table(ltfilename)
                else:
                    self.make_lookup_table()
            self.is_open = True

        else:
            self.fp = None
            self.is_open = False

        return

    def open(self, filename, open_mode='r'):
        """ Opens the specified file.

        Parameters
        ----------
        filename : string
            The name of the file to open
        """

        if 'b' not in open_mode:
            open_mode += 'b'

        self.fp = open(filename, open_mode)
        self.is_open = True

        return

    def close(self):
        """Close an open file"""

        if self.is_open:
            self.fp.close()
            self.is_open = False

        return

    def _read(self):
        raise NotImplementedError()

    def make_lookup_table(self):
        raise NotImplementedError()

    def load_lookup_table(self, filename):
        """Read a lookup table from a file.

        The file should contain the (integer) file offsets, with one on each line.

        Parameters
        ----------
        filename : string
            The path to the file.
        """

        self.lookup = []

        try:
            file = open(filename, 'r')
            for line in file:
                self.lookup.append(int(line.rstrip()))
        except FileNotFoundError:
            print("File name was not valid")
            return

    def read_event_by_number(self, num):
        """Reads a specific event from the file, based on its event number.

        This function uses the lookup table to find the requested event in the file, and then reads and returns that
        event. The current_event index is updated to the event number that was requested.

        Parameters
        ----------
        num : int
            The event number to read. This should be an index in the bounds of :attr:`lookup`

        Returns
        -------
        Event
            The requested event from the file
        """

        if 0 <= num < len(self.lookup):
            self.current_event = num
            self.fp.seek(self.lookup[num])
            return self._read()
        else:
            print("The provided number is outside the range of event numbers.")
            return None

    def __len__(self):
        return len(self.lookup)

    def __iter__(self):
        """ Prepare the file object for iteration.

        This sets the current event to 0 and rewinds the file pointer back to the beginning.
        """
        self.fp.seek(self.lookup[0])
        self.current_event = 0
        return self

    def __next__(self):
        """ Get the next event when iterating.
        """

        if self.current_event < len(self.lookup):
            evt = self.read_event_by_number(self.current_event)
            self.current_event += 1
            return evt
        else:
            raise StopIteration

    def evtrange(self, start=0, stop=None, step=1):
        """Iterate over the given range of events.

        This method returns a generator object that returns the events from the file with the given range of event IDs.
        This is a better approach for files with large events, as it only reads each event as needed.

        Parameters
        ----------
        start : int, optional
            The event to start at. Defaults to the beginning of the file.
        stop : int, optional
            One plus the index of the last event to read. (This follows the Python slicing convention of not including
            the upper bound.) Defaults to the end of the file.
        step : int, optional
            The step between events. Defaults to 1.

        Yields
        ------
        Event
            An event from the file.

        Raises
        ------
        IndexError
            If `start` or `stop` was out-of-bounds.
        """

        if stop is None:
            stop = len(self.lookup)

        if start < 0 or stop > len(self.lookup):
            raise IndexError("The given start and/or stop were out of bounds")

        for i in range(start, stop, step):
            yield self.read_event_by_number(i)

    def __getitem__(self, item):
        """ Implements subscripting of the event file, with slices.
        """

        if isinstance(item, slice):
            start, stop, step = item.indices(len(self.lookup))
            evts = []
            for i in range(start, stop, step):
                e = self.read_event_by_number(i)
                evts.append(e)
            return evts

        else:
            if item < 0 or item > len(self.lookup):
                raise IndexError("The index is out of bounds")
            else:
                return self.read_event_by_number(item)


class Error(Exception):
    """Base class for custom exceptions."""
    pass


class FilePosError(Error):
    """Raised when data is read from a file whose read pointer does not seem
    to be in the correct position.

    Parameters
    ----------
    filename : string
        The name of the file
    details : string, optional
        More details about the error
    """

    def __init__(self, filename, details=""):
        self.filename = filename  #: The name of the file where the error occurred
        self.details = details    #: Details of the error
        self.message = "FilePosError in " + filename + ": " + details  # An error message for printing
        super().__init__()
