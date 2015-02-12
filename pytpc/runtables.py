"""
runtables
=========

This module provides functions for getting run data from a CSV file.

"""

from __future__ import division, print_function
import csv

cleanmap = {ord(' '): '_',
            ord(':'): None}


def _sanitize(s):
    """Sanitizes the string s"""
    return s.split('(')[0].rstrip().lower().translate(cleanmap)


def parse(csvdata):
    """Parse the given file of run tables.

    The input file should be in CSV format. The first column of the file is assumed to contain a run number, and
    the rest of the columns are arbitrary. The first row should contain headers labeling the fields. The headers
    will be used as keys in the resulting dictionary after they have been sanitized.

    Sanitization of column headers consists of the following

        - Remove everything after a parenthesis (to eliminate unit labels)
        - Convert everything to lowercase
        - Convert spaces to underscores

    This is accomplished in the private function _sanitize.

    **Parameters**

    csvdata : iterable
        An iterable (such as an open file) that contains the run tables in a format that can be parsed by the
        Python CSV module.

    **Returns**

    rundata : dict
        A dictionary with the run number as a key, and another dictionary as the value. The embedded dictionary
        has a key:value pair for each column in the run tables.

    """
    # Load the CSV runtables
    cr = csv.reader(csvdata)
    raw = []
    for row in cr:
        raw.append(row)

    rtkeys = list(map(_sanitize, raw[0]))  # The keys for the dictionary of run info
    raw.pop(0)

    # Make a dictionary of run data. Assume run rumber in first column
    rundata = {}
    for row in raw[1:]:
        n = int(row[0])
        d = {k: v for k, v in zip(rtkeys, row)}
        rundata[n] = d

    return rundata