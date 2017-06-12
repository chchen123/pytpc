Preliminary Steps
=================

1) Install Python 3.6+
----------------------
There are binaries for macOS, but you will need to compile it for Linux if you don't have root access.
...
...
...


2) Install virtualenvwrapper
----------------------------


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

CMake_ is am open source software to control the worlkflow and build process of software. To install CMake, enter the following command into the command line (using a package manager of your choice):

.. code-block:: shell

   sudo apt-get install cmake

After installation you can check the version of CMake that was installed with the following call:

.. code-block:: shell

   cmake --version

.. _CMake: https://cmake.org/

.. warning:: 
   Depending on the OS being run, a repository update may be neccesary to install the newest version of these dependencies.

b) Install Armadillo
********************

Armadillo_ is a linear algebra library for C++. To install Armadillo, it is best to follow the instructions outlined `here <http://arma.sourceforge.net/download.html>`__. First, install the reccomended packages based on your OS. Then, in the command line, use wget or an equivalent to download the .tar archive and extract it:

.. code-block:: shell
   
   wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
   tar xf armadillo-7.950.1.tar.xz

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

.. _Armadillo: http://arma.sourceforge.net/


c) Install the HDF5 Library
***************************

The `HDF5 Library`_ (compiled with C++ support) is used for storing and managing raw experimental data. It is easiest install and build the library with CMake, the steps for which can be `here <https://support.hdfgroup.org/HDF5/release/cmakebuild518.html>`__. To download and uncompress the file, find the correct file link and enter the following into the command line.

.. code-block:: shell

   wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.18.tar.gz 
   tar xf CMake-hdf5-1.8.18.tar.gz 

Then change to the directory created by the extraction and excecute the batch file named *build-unix.sh*.

.. code-block:: shell

   ./batch-shell.sh

This will place the built binary in the bin folder and run through an extensive series of tests for correct installation.

.. _HDF5 Library: https://support.hdfgroup.org/HDF5/

d)Install and Compile mcopt Library
***********************************

Now install the mcopt library itself; it can be found `here <https://help.ubuntu.com/lts/serverguide/git.html>`__. Install the repsitory locally using the .git link found on github.

.. code-block:: shell

   git clone https://github.com/jbradt/mcopt.git

Then follow the instructions in the README.md file found in the mcopt directory.
run tests


4) Create a new Virtual Env
---------------------------


5) Install the pytpc Package
----------------------------


6) Create a Config File
-----------------------
Create a config file for the analysis code. There is a template on the ______ page of this sphinx documentation or use the one created for argon-46 which can be found here_. 

.. _here: https://github.com/jbradt/ar40-aug15/blob/master/fitters/config_e15503b.yml

7) Set Up Energy Loss Data
--------------------------

