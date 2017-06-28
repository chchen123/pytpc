Preliminary Steps for macOS
===========================

1) Install Python 3.6+
----------------------
Check your current python version(s) by entering the following commands into the command line: 


.. code-block:: shell

   python --version
   python3 --version
   python3.6 --version

Running **macOS** install/update Python 3.0+ (here using the reccomended `Homebrew <https://brew.sh/>`__ package manager):

.. code-block:: shell

   brew install python3
   brew upgrade python3 

A .profile file is a shell script that Bash runs whenever it is started or executed by the user. The installation instructions for Linux distributions involve adding lines to a built-in .bashrc, and the .profile is the macOS equivalent. However, you must first create the .profile file as it is not included on Mac systems by default. Create and open a plain text file titled `.profile.txt` in your home directory. This file will not appear in your home directory unless you enter 'ls -a' in the terminal to show hidden files.

To set these changes into effect either restart your shell or enter the following command in the command line.

.. code-block:: shell
      
   source ~/.profile


2) Install Clang and virtualenvwrapper
--------------------------------------

a) Install/Update Clang
***********************

`Clang <https://clang.llvm.org/>`__ is a compiler front end that supports multiple languages. Although an older version is already installed on your system, install the newest version using brew:

.. code-block:: shell

   brew install llvm

To make this most recent version of clang the default compiler on your system (and enable OpenMP support) paste the following lines into the .profile file created above:

.. code-block:: shell

   export CC=/usr/local/Cellar/gcc/7.1.0/bin/gcc-7
   export CXX=/usr/local/Cellar/gcc/7.1.0/bin/g++-7
   export LDFLAGS=-L$(brew --prefix llvm)/lib

OpenMP is packaged with the new versions of the clang compiler. This tool allows for shared memory multiprocessing in C and C++; in the context of this software OpenMP allows for parallel track generation during the minimization and is highly reccomended for running the analysis.

b) Install virtualenvwrapper
****************************

Install and setup virtualenvwrapper and associated tools. This allows you to create isolated "virtual environments" with independent installations of Python packages. This isn't strictly necessary, but helps prevent conflicts between incompatible package versions. To install virtualenvwrapper, run the command below:

.. code-block:: shell
   
   pip3 install virtualenvwrapper   # sudo might be required

An introduction and walkthrough to using the virtualenvwrapper tool can be found `here <https://virtualenvwrapper.readthedocs.io/en/latest/>`__. To use virtual environments, place the following lines in the .profileF file created in the previous step. The first line sets the Python interpreter for your virtual environments to python3.6. The last line is a path to your shell startup file and you should change it depending on where virtualenvwrapper was installed by pip.

.. code-block:: shell

   VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3.6
   export WORKON_HOME=$HOME/.virtualenvs
   export MSYS_HOME=/c/msys/1.0
   source /usr/local/bin/virtualenvwrapper.sh


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

`CMake <https://cmake.org/>`__ is an open-source software that controls the workflow and build process of software. Install CMake using brew:

.. code-block:: shell

   brew install cmake

b) Install Armadillo
********************

`Armadillo <http://arma.sourceforge.net/>`__ is a wrapper that presents a clean interface to several linear algebra libraries. Install Armadillo using brew (this requires the addition of a Homebrew/science repository):

.. code-block:: shell
   
   brew tap homebrew/science
   brew install armadillo

c) Install the HDF5 Library
***************************

The `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__ (compiled with C++ support) is used for storing and managing raw experimental data. Brew most likely installed the HDF5 Library as a dependency for armadillo, but run the following command to be sure:

.. code-block:: shell

   brew install hdf5 

d) Install and Compile mcopt Library
************************************

Finally, install the mcopt library itself; it can be found `here <https://github.com/jbradt/mcopt>`__. Install the repository locally using the .git link found on GitHub.

.. code-block:: shell

   git clone https://github.com/jbradt/mcopt.git
   cd mcopt

The compilation and installation instructions can be found in the README.md file in this directory. The necessary commands are as follows.

.. code-block:: shell

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   make install   # sudo might be required

.. note::

   Other flags may be neccesary depending on where the mcopt libary is to be installed. Refer to the `CMake documentation <https://cmake.org/cmake/help/v3.9/index.html#>`__ for information on this.

Test for correct code compilation by executing the *test_mcopt* file:

.. code-block:: shell

   ./test_mcopt


4) Create a new Virtual Env
---------------------------

Now, create a virtual environment by entering the following into the command line:

.. code-block:: shell

   mkvirtualenv [name]

Refer to the link in step 2 for information on using and managing virtual environments.


5) Install the pytpc Package
----------------------------

Now, install the pytpc package and its dependencies; it can be found `here <https://github.com/ATTPC/pytpc.git>`__. Install the repository locally using the .git link found on GitHub.

.. code-block:: shell

   git clone https://github.com/ATTPC/pytpc.git
   cd pytpc

Installation instructions can be found in the README.md file. Use pip to manage the required Python software packages.

.. code-block:: shell

   pip3 install Cython numpy scipy sklearn scikit-learn matplotlib seaborn jinja2 pandas clint pyYaml sqlalchemy tables h5py sphinx   # sudo might be required

Then, to install pytpc from the source code, run:

.. code-block:: shell

   python3 setup.py install

To test for correct installation. Run the provided tests with the following commands (not all tests print output to the screen but none should throw errors):

.. code-block:: shell
   
   cd pytpc/tests
   python3.6 -m unittest discover


6) Create a Config File
-----------------------

Create a config file for the analysis code. There is a template in the next section of this documentation.


7) Set Up Energy Loss Data
--------------------------
Set up the energy loss info for the relevant nuclei.


*Tested for macOS Sierra*
