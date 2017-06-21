1) Merge and Unpack GRAW Files into HDF5 Files
==============================================

This step in the analysis should be performed locally. Install the graw-merger tool and it's dependencies. The graw-merger repository contains the source code for the `graw2hdf` tool, which can be used to merge the GRAW files produced by the GET electronics into an `HDF5` <https://www.hdfgroup.org/HDF5/>`__ file. Armadillo should already be installed, but some `Boost C++ libraries <http://www.boost.org/>`__ are are required for this step. Use wget to download to download the .tar archive (vesion 1.55 or later) and extract it (find the correct download `here <https://dl.bintray.com/boostorg/release/1.64.0/source/>`__):

.. code-block:: shell

   wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
   tar xzf boost_1_64_0.tar.gz
   cd boost_1_64_0

See the index.html file in the current directory for more information on the current boost version; the "Getting Started" section (5) contains useful installation instructions, but the neccesary procedure is outlined here. Here we build all Boost libraries for simplicities sake, see graw-merger's readme for the specific libraries necesary:

.. code-block:: shell 

   ./bootstrap.sh
   ./b2 install


Now, clone the graw-merger repository from GitHub which, can be found `here <https://github.com/ATTPC/graw-merger>`__:

.. code-block:: shell

   git clone https://github.com/ATTPC/graw-merger.git
   cd graw-merge

More information on installation and compilation can be found in the README.md file packaged with the software, but the instructions are outlined here.
