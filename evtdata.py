import struct
import numpy
import os.path
from tpcplot import generate_pad_plane


class EventFile:
    """Represents a merged GET event file.

    This class provides an interface to merged event files from the GET electronics. It is capable of opening
    merged data files and reading events from them.
    """

    def __init__(self, filename=None):
        """ Initializes and, optionally, opens an event file.

        If a filename is provided, that file will be opened.

        :param filename: The name of the file to open.
        """

        self.magic = 0x6e7ef11e
        self.lookup = []
        self.current_event = 0

        if filename is not None:
            self.open(filename)
            ltfilename = os.path.splitext(filename)[0] + '.lookup'
            if os.path.exists(ltfilename):
                self.load_lookup_table(ltfilename)
            else:
                self.make_lookup_table()
            self.is_open = True

        else:
            self.fp = None
            self.is_open = False

        return

    def open(self, filename):
        """ Opens the specified file.

        The file's first 4 bytes are compared to the magic value ``0x6e7ef11e`` to check that the file is of the
        correct type.

        :param filename: The name of the file to open
        :except: IOError
        """

        try:
            self.fp = open(filename, 'rb')
            self.is_open = True
            print("Opened file: " + filename)

            read_magic = struct.unpack('<I', self.fp.read(4))[0]
            if read_magic != self.magic:
                print("Bad file. Magic number is wrong.")

        except IOError:
            raise

        return

    def close(self):
        """Close an open file"""

        if self.is_open:
            self.fp.close()
            self.is_open = False

        return

    @staticmethod
    def pack_sample(tb, val):
        """Pack a sample value into three bytes.

        See the documentation of the Event File format for details.

        :param tb: The time bucket
        :param val: The sample
        :return: The packed data
        """

        # val is 12-bits and tb is 9 bits. Fit this in 24 bits.
        # Use one bit for parity

        if val < 0:
            parity = 1 << 12
            narrowed = -val
        elif val >= 4095:
            parity = 0
            narrowed = 4095
        else:
            parity = 0
            narrowed = val

        joined = (tb << 15) | narrowed | parity
        return joined

    @staticmethod
    def unpack_sample(packed):
        """Unpacks a packed time bucket / sample pair.

        :param packed: The packed value, as an integer
        :return: A tuple containing (time bucket, sample)
        """

        tb = (packed & 0xFF8000) >> 15
        sample = (packed & 0xFFF)
        parity = (packed & 0x1000) >> 12

        if parity == 1:
            sample *= -1

        return tb, sample

    @staticmethod
    def unpack_samples(packed):
        """ Unpacks an array of time bucket / sample pairs.

        This is really just a modified version of unpack_sample that works with arrays.

        :param packed: An array of packed values.
        :return: An array of sample values, indexed by time bucket.
        """

        # TODO: Merge this into the unpack_sample method?

        tbs = (packed & 0xFF8000) >> 15
        samples = (packed & 0xFFF)
        parities = (packed & 0x1000) >> 12

        # Change the parities from (0, 1, ...) to (1, -1, ...) and multiply
        samples *= (-2 * parities + 1)

        unpacked = numpy.hstack((tbs, samples))
        result = numpy.zeros(512)

        for item in unpacked:
            result[item[0]] = item[1]

        return result

    def _read(self):
        """Reads an event at the current file position.

        This function attempts to read an event beginning at the current file position. The position is checked by
        checking the event magic number ``0xEE``.

        :return: The event as an :class:`Event`
        """

        assert self.is_open

        try:
            # Read the event header
            # This has the following structure:
            #     (magic, size, ID, ts, nTraces)
            hdr = struct.unpack('<BIIQH', self.fp.read(19))

            if hdr[0] != 0xEE:
                # This is not the beginning of the event. Seek back and fail.
                self.fp.seek(-19, 1)
                raise FilePosError(self.fp.name, "Event magic number was wrong.")

            event_size = hdr[1]
            event_id = hdr[2]
            event_ts = hdr[3]
            num_traces = hdr[4]

            print("Found event of size", event_size, ":")
            print("    Event Number:", event_id)
            print("    Contains", num_traces, "traces")

            new_evt = Event(evt_id=event_id, timestamp=event_ts)
            new_evt.traces = numpy.zeros((num_traces,), dtype=new_evt.dt)

            # Read the traces

            for n in range(num_traces):
                # Read the trace header. The structure of this is:
                #    (size, cobo, asad, aget, ch, pad)
                th = struct.unpack('<IBBBBH', self.fp.read(10))

                tr = new_evt.traces[n]

                tr['cobo'] = th[1]
                tr['asad'] = th[2]
                tr['aget'] = th[3]
                tr['channel'] = th[4]
                tr['pad'] = th[5]

                # Now read the trace itself
                num_samples = (th[0] - 10) // 3  # (total - header) / size of packed item

                packed = numpy.hstack((numpy.fromfile(self.fp, dtype='3u1', count=num_samples),
                                       numpy.zeros((num_samples, 1), dtype='u1'))).view('<u4')

                unpacked = self.unpack_samples(packed)
                tr['data'][:] = unpacked

            return new_evt

        except Error as er:
            raise er

    def make_lookup_table(self):
        """ Indexes the open file and generates a lookup table.

        The lookup table is a list of the file offsets of each event, in bytes. This is used by the read_next() and
        read_previous() functions to navigate around the file. The table is returned, but it is also assigned to
        self.lookup.

        :return: The lookup table.
        """

        assert self.is_open

        self.fp.seek(4)

        self.lookup = []

        while True:
            magic = self.fp.read(1)
            if magic == b'\xEE':
                pos = self.fp.tell() - 1
                self.lookup.append(pos)
                offset = struct.unpack('<I', self.fp.read(4))[0]
                self.fp.seek(pos + offset)
            elif magic == b'':
                break
            else:
                print("Bad magic number")
                return None

        self.fp.seek(4)
        self.current_event = 0

        ltfilename = os.path.splitext(self.fp.name)[0] + '.lookup'

        if not os.path.exists(ltfilename):
            ltfile = open(ltfilename, 'w')
            for entry in self.lookup:
                ltfile.write(str(entry) + '\n')
            ltfile.close()

        return self.lookup

    def load_lookup_table(self, filename):
        """Read a lookup table from a file.

        The file should contain the (integer) file offsets, with one on each line.

        :param filename: The path to the file.
        """

        self.lookup = []

        try:
            file = open(filename, 'r')
            for line in file:
                self.lookup.append(int(line.rstrip()))
        except FileNotFoundError:
            print("File name was not valid")
            return

    def read_next(self):
        """Read the next event in the file.

        This function reads the next event from the file and increments the self.current_event value. For example,
        if the current_event is 4, then current_event will be set to 5, and event 5 will be read and returned.

        :return: The next event, as an Event object.
        """

        if self.current_event + 1 < len(self.lookup):
            self.current_event += 1
            self.fp.seek(self.lookup[self.current_event])
            return self._read()
        else:
            print("At last event")
            return None

    def read_current(self):
        """Read the event currently pointed to by the current_event marker.

        This function reads the current event from the file. For example, if the current_event is 4, then event 4
        will be read and returned.

        :return: The current event, as an Event object.
        """

        self.fp.seek(self.lookup[self.current_event])
        return self._read()

    def read_previous(self):
        """Read the previous event from the file.

        This function reads the previous event from the file and decrements the self.current_event value. For example,
        if the current_event is 4, then current_event will be set to 3, and event 3 will be read and returned.

        :return: The previous event, as an Event object.
        """

        if self.current_event - 1 >= 0:
            self.current_event -= 1
            self.fp.seek(self.lookup[self.current_event])
            return self._read()
        else:
            print("At first event")
            return None

    def read_event_by_number(self, num):
        """Reads a specific event from the file, based on its event number.

        This function uses the lookup table to find the requested event in the file, and then reads and returns that
        event. The current_event index is updated to the event number that was requested.

        :param num: The event number to read.
        :return: The requested event, as an Event object.
        """

        if 0 <= num < len(self.lookup):
            self.current_event = num
            self.fp.seek(self.lookup[num])
            return self._read()
        else:
            print("The provided number is outside the range of event numbers.")
            return None


class Event:
    """ Represents an event from an :class:`EventFile`.

    The event is represented as a numpy array. The indices are:
        (CoBo, AsAd, AGET, channel, time bucket)
    """

    def __init__(self, evt_id=0, timestamp=0):
        """ Initialize a new event.

        :param evt_id: The event ID
        :param timestamp: The event timestamp
        :return: None
        """
        assert (evt_id >= 0)
        assert (timestamp >= 0)

        self.dt = numpy.dtype([('cobo', 'u1'), ('asad', 'u1'), ('aget', 'u1'), ('channel', 'u1'),
                                    ('pad', 'u2'), ('data', '512i2')])

        self.evt_id = evt_id
        self.timestamp = timestamp
        self.traces = numpy.zeros((0,), dtype=self.dt)

        return

    def hits(self):
        """ Calculate the total activation of each pad.

        The result is a numpy array of (pad_number, activation)

        :return: A numpy array of (pad_number, activation)
        """

        hits = self.traces['data'].sum(1)
        pads = self.traces['pad']

        flat_hits = numpy.zeros(10240)
        for (p, h) in zip(pads, hits):
            flat_hits[p] = h

        return flat_hits

    def xyzs(self):
        """ Find the scatter points of the event in space.

        The result is a 4-D array. The points are given as (x, y, time bucket, activation). The x and y coordinates
        are in millimeters.

        :return: A numpy array of (x, y, tb, activation)
        """

        nz = self.traces['data'].nonzero()
        pcenters = generate_pad_plane().mean(1)

        xys = numpy.array([pcenters[self.traces[i]['pad']] for i in nz[0]])
        zs = nz[1].reshape(nz[1].shape[0], 1)
        cs = self.traces['data'][nz].reshape(nz[0].shape[0], 1)

        result = numpy.hstack((xys, zs, cs))
        return result


def load_pedestals(filename):
    """ Loads pedestals from the provided file.

    :param filename: The name of the file containing the pedestals.
    :return: The pedestals as a numpy array
    """

    f = open(filename, 'r')
    peds = numpy.zeros((10, 4, 4, 68, 512))

    for line in f:
        subs = line[:-1].split(',')  # throw out the newline and split at commas
        vals = [int(float(x)) for x in subs]
        peds[(vals[0], vals[1], vals[2], vals[3])] = vals[4] * numpy.ones(512)

    return peds


def load_padmap(filename):
    """ Loads pad mapping from the provided file.

    :param filename: The name of the file containing the pad mapping.
    :return: The pad mapping as a dictionary {(cobo, asad, aget, ch): pad}
    """

    f = open(filename, 'r')
    pm = {}

    for line in f:
        subs = line[:-1].split(',')  # throw out the newline and split at commas
        vals = [int(float(x)) for x in subs]
        pm[(vals[0], vals[1], vals[2], vals[3])] = vals[4]

    return pm


class Error(Exception):
    """Base class for custom exceptions."""
    pass


class FilePosError(Error):
    """Raised when data is read from a file whose read pointer does not seem
    to be in the correct position.

    Attributes:
        filename -- the name of the file being read
        details -- an explanation of the particular error
        message -- an error message for printing
    """

    def __init__(self, filename, details=""):
        self.filename = filename
        self.details = details
        self.message = "FilePosError in " + filename + ": " + details