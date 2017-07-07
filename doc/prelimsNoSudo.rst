Preliminary Steps Without Sudo Privileges
=========================================

If you plan to run the analysis on a system that you do not have sudo privileges on, such as an HPC, you must install the dependencies and target packages locally and add their locations to your path. In this case, setup procedures will differ by system based on what is already available. Below is a list providing dependencies and the software itself.

`Python 3.6+ <https://www.python.org/downloads/>`__
   - `virtualenvwrapper <http://virtualenvwrapper.readthedocs.io/en/latest/>`__ (recommended but not required)


C++ Compiler
   - Be sure that you are using a compiler with openMP support (GNU is recommended).


`mcopt Library <https://github.com/jbradt/mcopt>`__
   - `CMake <https://cmake.org/>`__
   - `Armadillo <http://arma.sourceforge.net/>`__
   - `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__


`pytpc Package <https://github.com/ATTPC/pytpc.git>`__
   - Refer to the `requirements.txt` file in pytpc for the required Python packages and version numbers. Pip is recommended for installing packages, but not required.


`graw-merger Tool <https://github.com/ATTPC/graw-merger>`__
   - `Boost C++ Libraries <http://www.boost.org/>`__ (version 1.55 or later)
