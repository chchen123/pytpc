3) Clean the GET Data with the Hough Transform Code
===================================================

Implementation
--------------
**This step in the analysis should be performed on an on an HPC.** The script for the `pyclean` Hough transform cleaning code is in the main pytpc repository (pytpc/bin/pyclean) and the implementation in the Python module `pytpc.cleaning`. The `pyclean` script parses through a data file and removes noise.

This code:

#. Does a nearest neighbor comparison to eliminate statistical noise
#. Does a circular Hough transform to find the center of the spiral or curved track in the micromegas pad plane
#. Does a linear Hough transform on (z,r*phi) to find which points lie along the spiral/curve
#. Writes points and their distance from the line to an HDF5 file

The "Cleaning parameters" in the config file decide how aggressively the data is cleaned.

Usage
-----

`pyclean` can be used as follows:

.. code-block:: shell

   pyclean [-h] [--canon-evtids CANON_EVTIDS] config input output

The `config` argument takes the path to the proper config file. The `input` argument takes the path to the HDF5 file to be cleaned. The `output` argument takes the path to where the cleaned HDF5 file will be written and requires a filename such as clean_run_XXXX.h5 to create a file by that name. If no path, only a filename, is provided, the file will be created in the same directory as the original HDF5. The optional `CANON_EVTIDS` arguments provides a path to an HDF5 file containing canonical evt ids. 

The following command will display this information in the terminal:

.. code-block:: shell

   pyclean -h

