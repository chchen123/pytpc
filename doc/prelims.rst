Preliminary Steps
=================

Steps differ between operating systems, so be sure to follow the correct instructions for your system.

1) Install Python 3.6+
----------------------
Check your current python version(s) by entering the following commands into the command line: 


.. code-block:: shell

   python --version
   python3 --version
   python3.6 --version

If Python 3.6+ is not present, follow these instructions depending on your operating system:

Running a **Linux distribution**, Python must be compiled. Download and compile the most recent Python release with the following commands (find the link to the correct .tar file `here <https://www.python.org/downloads/>`__):

.. note::

   If CMake is not already on your machine see step 3a to install it.

.. code-block:: shell
   
   wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tar.xz
   tar xvf Python-3.6.1.tar.xz   # sudo might be required
   cd Python-3.6.1
   ./configure --with-ensurepip=install   # sudo might be required
   make
   make test
   make install   # sudo might be required

Some **newer Linux distributions** may be able to install Python 3.6+ using a package manager:

.. code-block:: shell

   apt-get install python3.6   # sudo might be required

Running **macOS** install/update Python 3.0+ (here using the reccomended `Homebrew <https://brew.sh/>`__ package manager):

.. code-block:: shell

   brew install python3
   brew upgrade python3 

A .profile file is a shell script that Bash runs whenever it is started or executed by the user. Setting up pytpc on Linux distributions involves adding lines to a built-in .bashrc, and the .profile is the macOS equivalent. However, you must first create the .profile file as it is not included on Mac systems by default. **Create and open a plain text file titled `.profile` in your home directory**. This file will not appear in your home directory unless you enter 'ls -a' in the terminal to show hidden files.

.. note::

   Remember to use the correct call to python/python3/python3.6 and pip/pip3/pip3.6 depending how your Python 3.6 was installed. This is neccesary to install software in the correct locations for the remainder of these instructions.


2) Install Clang and virtualenvwrapper
--------------------------------------

a) Install/Update Clang (macOS Only)
************************************

`Clang <https://clang.llvm.org/>`__ is a compiler front-end that supports multiple languages. Although an older version is already installed on your system, install the newest version using brew:

.. code-block:: shell

   brew install llvm

To make this most recent version of clang the default compiler on your system (and enable OpenMP support) paste the following lines into the .profile file created above:

.. code-block:: shell

   LLVM_PATH=$(brew --prefix llvm)
   export CC=${LLVM_PATH}/bin/clang-4.0
   export CXX=${LLVM_PATH}/bin/clang++
   export LDFLAGS=-L${LLVM_PATH}/lib

OpenMP is packaged with the new versions of the clang compiler. This tool allows for shared memory multiprocessing in C and C++; in the context of this software OpenMP allows for parallel track generation during the minimization and is highly reccomended for running the analyses.

b) Install virtualenvwrapper
****************************

Install and setup virtualenvwrapper and associated tools. This allows you to create isolated "virtual environments" with independent installations of Python packages. This isn't strictly necessary, but helps prevent conflicts between incompatible package versions. To install virtualenvwrapper, run the command below:

.. code-block:: shell
   
   pip install virtualenvwrapper   # sudo might be required

An introduction and walkthrough to using the virtualenvwrapper tool can be found `here <https://virtualenvwrapper.readthedocs.io/en/latest/>`__. To use virtual environments, place the following lines in your .bashrc or .profile file. The first line sets the Python interpreter for your virtual environments to python3.6. The last line is a path to your shell startup file, and you should change it depending on where virtualenvwrapper was installed by pip.

.. code-block:: shell

   VIRTUALENVWRAPPER_PYTHON=$(which python3.6)
   export WORKON_HOME=$HOME/.virtualenvs
   source /usr/local/bin/virtualenvwrapper.sh

.. note:: 
   
   Remember to enter `source ~/.basrhc` or `source ~/.profile` into the command line or restart it so that these changes take effect.


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

`CMake <https://cmake.org/>`__ is an open-source software that controls the workflow and build process of software. To install CMake, enter the following command into the command line using a package manager of your choice (Homebrew if using macOS):

.. code-block:: shell

   apt-get install cmake   # sudo might be required

After installation, check the version of CMake that was installed with the following call:

.. code-block:: shell

   cmake --version

b) Install Armadillo
********************

`Armadillo <http://arma.sourceforge.net/>`__ is a wrapper that presents a clean interface to several linear algebra libraries. 

If running a **Linux distribution**, it is best to compile Armadillo from source following the instructions `here <http://arma.sourceforge.net/download.html>`__. The process is outlined below. First, install the reccomended packages based the OS being run. Then, in the command line, use wget to download the .tar archive and extract it (use the link above to find the most recent release):

.. code-block:: shell
   
   wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
   tar xzf armadillo-7.950.1.tar.xz   # sudo might be required
   cd armadillo-7.950.1
   cmake .
   make
   make install   # sudo might be required

To test that armadillo and its prerequisites have been installed correctly, run the compiled tester with the following commands:

.. code-block:: shell

   cd tests
   make clean
   make
   ./main

If running **macOS**, install Armadillo using brew (this requires the addition of a Homebrew/science repository):

.. code-block:: shell
   
   brew tap homebrew/science
   brew install armadillo

c) Install the HDF5 Library
***************************

The `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__ (compiled with C++ support) is used for storing and managing raw experimental data. 

If running a **Linux distribution**, it is easiest to install and build the library with CMake, the steps for which can be found `here <https://support.hdfgroup.org/HDF5/release/cmakebuild518.html>`__. To download and uncompress the file, find the link to the most recent release and enter the following into the command line.

.. code-block:: shell

   wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.19.tar.gz
   tar xzf CMake-hdf5-1.8.19.tar.gz   # sudo might be required
   cd CMake-hdf5-1.8.19
   ./build-unix.sh   # sudo might be required

This will place the built binary in the bin folder and run through a series of tests for correct installation.

If running **macoS**, Homebrew most likely installed the HDF5 Library as a dependency for armadillo, but run the following command to be sure:

.. code-block:: shell

   brew install hdf5 

d) Install and Compile mcopt Library
************************************

Finally, install the mcopt library itself; it can be found `here <https://github.com/jbradt/mcopt>`__. Clone the repository locally using the .git link from GitHub.

.. code-block:: shell

   git clone https://github.com/jbradt/mcopt.git
   cd mcopt

The compilation and installation instructions can be found in the README.md file in this directory. The necessary commands are as follows:

.. code-block:: shell

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   make install   # sudo might be required

.. note::

   Other flags may be neccesary depending on where the mcopt libary is to be installed. Refer to the `CMake documentation <https://cmake.org/cmake/help/v3.9/index.html#>`__ for information on this.

Test for correct code compilation by running the *test_mcopt* executable:

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

   pip install -r requirements.txt   # sudo might be required

Then, to install pytpc from the source code, run:

.. code-block:: shell

   python setup.py install   # sudo may be required

To test for correct installation, run the provided tests with the following command:

.. code-block:: shell
   
   python -m unittest discover


6) Create a Config File
-----------------------

Create a config file for the analysis code. There is an annotated template in the next section of this documentation.


7) Set Up Energy Loss Data
--------------------------
Set up the energy loss info for the relevant nuclei.


*Tested for Ubuntu 14.04 and 16.04 and macOS Sierra*
