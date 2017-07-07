Preliminary Steps Without Sudo Privileges
=========================================

If you plan to run the analysis using a system on which you do not have sudo or root privileges, local installations and path additions will be neccesary. In this case, setup procedures will differ by system based on what is already available. Below is a list to provide the dependencies for this software.

`Python 3.6+ <https://www.python.org/downloads/>`__
   - `virtualenvwrapper <http://virtualenvwrapper.readthedocs.io/en/latest/>`__


Compiler
   - Be sure that you are using a compiler with openMP support (GNU is reccomended). 


`mcopt Library <https://github.com/jbradt/mcopt>`__
   - `CMake <https://cmake.org/>`__
   - `Armadillo <http://arma.sourceforge.net/>`__
   - `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__


`pytpc Library <https://github.com/ATTPC/pytpc.git>`__
   - Required python packages are installed using pip. Refer to the `requirements.txt` file for specific packages and version numbers.


`graw-merger Library <https://github.com/ATTPC/graw-merger>`__
   - `Boost C++ Libraries <http://www.boost.org/>`__ (version 1.55 or later)
