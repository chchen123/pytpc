Preliminary Steps for macOS
===========================

1) Install Python 3.6+
----------------------
Check your current python version(s) by entering the following commands into the command line: 

.. code-block:: shell

   python --version
   python3 --version
   python3.6 --version

Running **macOS** install/update Python 3.0+ (here using the `Homebrew <https://brew.sh/>`__ package manager):

.. code-block:: shell

   brew install python3
   brew upgrade python3 

A .bashrc file is a shell script that Bash runs whenever it is started or executed by the user. The installation instructions for Linux distributions involve adding lines to the built-in .bashrc folder, and this step is neccesary for macOS for aliases and path adjustments. However, you must first create the a .bashrc file as it is not included by default on Mac systems. To create and open a plaintext .bashrc file enter the following into your command line:

.. code-block:: shell
   
   touch ~/.bashrc
   open ~/.bashrc

This file will not appear in your home directory unless a '-a' flag is included with an 'ls' command in the terminal. Although not strictly neccesary, paste the two lines below into your new .bashrc file; these lines map the python and pip commands to their Python 3.6+ counterparts (i.e. 
*python helloworld.py* will be executed using Python 3.6+).

.. code-block:: shell
      
   alias python=python3
   alias pip=pip3

To force set these changes into effect, enter the following command in the command line. You may also wish to have the command below run automatically on the startup of your shell. To do this paste the command into the the *Startup - Run command* field in Terminal>Preferences>Profiles>Shell.

.. code-block:: shell
      
   source ~/.bashrc


2) Install virtualenvwrapper
----------------------------
Install and setup virtualenvwrapper and associated tools. This allows you to create isolated "virtual environments with independent installations of Python packages. This isn't strictly necessary, but helps prevent conflicts between incompatible package versions.


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

`CMake <https://cmake.org/>`__ is an open-source software that controls the workflow and build process of software. To install CMake, enter the following command into the command line (using a package manager of your choice):

.. code-block:: shell

   sudo apt-get install cmake

After installation, check the version of CMake that was installed with the following call:

.. code-block:: shell

   cmake --version

.. warning:: 

   Depending on the OS being run, a repository update may be neccesary to install the newest version of CMake.

b) Install Armadillo
********************

`Armadillo <http://arma.sourceforge.net/>`__ is a wrapper that presents a clearn interface to several linear algebra libraries. To install Armadillo, it is best to follow the instructions outlined `here <http://arma.sourceforge.net/download.html>`__. First, install the reccomended packages based the OS being run. Then, in the command line, use wget or an equivalent to download the .tar archive and extract it (use the link above to find the most recent release):

.. code-block:: shell
   
   wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
   tar xzf armadillo-7.950.1.tar.xz

The README.txt file found in the folder created by unpacking the Armadillo archive contains the remaining instructions. The most important of these is to change to the directory of the created folder and enter the following commands to configure armadillo:

.. code-block:: shell

   cmake .
   make
   make install

To test that armadillo and its prerequisites have been installed correctly, run the included tester executable with the following commands:

.. code-block:: shell

   cd tests
   make clean
   make
   ./main

c) Install the HDF5 Library
***************************

The `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__ (compiled with C++ support) is used for storing and managing raw experimental data. It is easiest to install and build the library with CMake, the steps for which can be found `here <https://support.hdfgroup.org/HDF5/release/cmakebuild518.html>`__. To download and uncompress the file, find the link to the most recent release and enter the following into the command line.

.. code-block:: shell

   wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.19.tar.gz
   tar xzf CMake-hdf5-1.8.19.tar.gz 

Then change to the directory created by the extraction and execute the batch file named *build-unix.sh*.

.. code-block:: shell

   cd CMake-hdf5-1.8.19
   ./build-unix.sh

This will place the built binary in the bin folder and run through a series of tests for correct installation.

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
   sudo make install

Test for correct installation by executing the *test_mcopt* file:

.. code-block:: shell

   ./test_mcopt

4) Create a new Virtual Env
---------------------------


5) Install the pytpc Package
----------------------------

Now, install the pytpc package and its dependencies; it can be found `here <https://github.com/ATTPC/pytpc.git>`__. Install the repository locally using the .git link found on GitHub.

.. code-block:: shell

   git clone https://github.com/ATTPC/pytpc.git
   cd pytpc

Installation instructions can be found in the README.md file. However, it is best to avoid Anaconda when using pytpc due to assorted problems with dependency versions and etc. Use pip to manage and the required Python software packages.

.. code-block:: shell

   sudo pip3.6 install Cython numpy scipy sklearn scikit-learn matplotlib seaborn jinja2 pandas clint pyYaml sqlalchemy tables h5py sphinx

Then, to install pytpc from the source code, run:

.. code-block:: shell

   python3.6 setup.py install

To test for correct installation. Run the provided tests with the following commands (not all tests print output to the screen but none should throw errors):

.. code-block:: shell
   
   cd pytpc/tests
   python3.6 test_evtdata.py
   python3.6 test_gases.py
   python3.6 test_grawdata.py
   python3.6 test_hdfdata.py
   python3.6 test_relativity.py
   python3.6 test_simulation.py
   python3.6 test_ukf.py
   python3.6 test_utilities.py

6) Create a Config File
-----------------------

Create a config file for the analysis code. There is a template in the next section of this documentation or use the one created for argon-40 which can be found `here <https://github.com/jbradt/ar40-aug15/blob/master/fitters/config_e15503b.yml>`__.


7) Set Up Energy Loss Data
--------------------------
Set up the energy loss info for the relevant nuclei.

SAMPLE .bashrc FILE


*Tested for macOS *
