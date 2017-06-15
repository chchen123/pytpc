Preliminary Steps
=================

1) Install Python 3.6+
----------------------
Check your current python version(s) by entering the following commands into the command line: 

.. code-block:: shell

   python --version
   python3 --version

If Python 3.6+ is not present, follow these instructions depending on your operating system:

Running **macOS** install/update Python 3.0+ (here using the `Homebrew <https://brew.sh/>`__ package manager): **ADD DEV PART**

.. code-block:: shell

   brew install python3
   brew upgrade python3

If running **Ubuntu**, Python must be compiled. Download and compile the most recent Python release with the following commands (find the link to the correct .tar file `here <https://www.python.org/downloads/>`__):

.. note::

   If CMake is not already on your machine skip to step 3a to install it.

.. code-block:: shell
   
   cd /usr/src
   wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tar.xz
   sudo tar xzf Python-3.6.1.tar.xz
   cd Python-3.6.1
   sudo ./configure
   sudo make altinstall

**Replace tarball part with below (to get python-dev -- unless 3.6+ neccesary...)**

If running **Ubuntu**, install the newest Python 3.0+ version and its -dev version using apt-get. If you are running Ubuntu 16.04 or above install python3.6 and python3.6-dev.

.. code-block:: shell

   sudo apt-get install python3
   sudo apt-get install python3-dev

.. note::

   It can be helpful to use an alias to aid in differentiation between Python releases. Place this into your ~/.bashrc or ~/.bash_aliases file using the following code (alias to python3 or python3.6 depending on your current OS and be sure to restart the command line):

   .. code-block:: shell
      
      alias python=python3


2) Install virtualenvwrapper
----------------------------


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

`CMake <https://cmake.org/>`__ is am open source software to control the worlkflow and build process of software. To install CMake, enter the following command into the command line (using a package manager of your choice):

.. code-block:: shell

   sudo apt-get install cmake

After installation, check the version of CMake that was installed with the following call:

.. code-block:: shell

   cmake --version

.. warning:: 

   Depending on the OS being run, a repository update may be neccesary to install the newest version of CMake.

b) Install Armadillo
********************

`Armadillo <http://arma.sourceforge.net/>`__ is a linear algebra library for C++. To install Armadillo, it is best to follow the instructions outlined `here <http://arma.sourceforge.net/download.html>`__. First, install the reccomended packages based the OS being run. Then, in the command line, use wget or an equivalent to download the .tar archive and extract it:

.. code-block:: shell
   
   wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
   tar xzf armadillo-7.950.1.tar.xz

The README.txt file found in the folder created by unpacking the armadillo archive contains the remaining instructions. The most important of these is to change to the directory of the created folder and enter the following commands to configure armadillo:

.. code-block:: shell

   cmake .
   make
   make install

To test that armadillo and its prerequisites have been installed correctly, change to the "/tests" directory and enter the commands:

.. code-block:: shell

   make clean
   make
   ./main

c) Install the HDF5 Library
***************************

The `HDF5 Library <https://support.hdfgroup.org/HDF5/>`__ (compiled with C++ support) is used for storing and managing raw experimental data. It is easiest install and build the library with CMake, the steps for which can be `here <https://support.hdfgroup.org/HDF5/release/cmakebuild518.html>`__. To download and uncompress the file, find the correct file link and enter the following into the command line.

.. code-block:: shell

   wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.18.tar.gz 
   tar xzf CMake-hdf5-1.8.18.tar.gz 

Then change to the directory created by the extraction and excecute the batch file named *build-unix.sh*.

.. code-block:: shell

   ./batch-shell.sh

This will place the built binary in the bin folder and run through an extensive series of tests for correct installation.

d) Install and Compile mcopt Library
************************************

Finally, install the mcopt library itself; it can be found `here <https://github.com/jbradt/mcopt>`__. Install the repsitory locally using the .git link found on github.

.. code-block:: shell

   git clone https://github.com/jbradt/mcopt.git
   cd mcopt

The compilation and installation instructions can be found in the README.md file in this directory. The neccesary commands are as follows.

.. code-block:: shell

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make
   sudo make install

Test for correct installation by excecuting the *test_mcopt* file:

.. code-block:: shell

   ./test_mcopt

4) Create a new Virtual Env
---------------------------


5) Install the pytpc Package
----------------------------

Now install the pytpc package and its dependencies; it can be found `here <https://github.com/ATTPC/pytpc.git>`__. Install the repository locally usign the .git link found on github.

.. code-block:: shell

   git clone https://github.com/ATTPC/pytpc.git
   cd pytpc

Installation instructions can be found in the README.md file. However, it is best to avoid Anaconda when using pytpc due to assorted problems with dependency versions and etc. Use pip to manage and the required Python software packages.

.. code-block:: shell

   sudo apt-get install python3-pip
   sudo python3 -m pip install --upgrade pip
   sudo pip3 install numpy scipy Cython scikit-learn matplotlib seaborn sphinx pyYaml sqlalchemy tables

Then, to install pytpc from the source code, run:

.. code-block:: shell

   python setup.py install

**Tests**

6) Create a Config File
-----------------------

Create a config file for the analysis code. There is a template on the *config* page of this sphinx documentation or use the one created for argon-46 which can be found `here <https://github.com/jbradt/ar40-aug15/blob/master/fitters/config_e15503b.yml>`__. The next section contains more informaton regarding the config file.


7) Set Up Energy Loss Data
--------------------------
Set up the energy loss info for the relevant nuclei.

*Tested for Ubuntu 14.04 and 16.04.*
