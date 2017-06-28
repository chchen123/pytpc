Preliminary Steps for Linux Distributions
=========================================

1) Install Python 3.6+
----------------------
Check your current python version(s) by entering the following commands into the command line: 

.. code-block:: shell

   python --version
   python3 --version
   python3.6 --version

If Python 3.6+ is not present, follow these instructions depending on your operating system:

Running a **Linux** distribution, Python must be compiled. Download and compile the most recent Python release with the following commands (find the link to the correct .tar file `here <https://www.python.org/downloads/>`__):

.. note::

   If CMake is not already on your machine see step 3a to install it.

.. code-block:: shell
   
   wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tar.xz
   sudo tar xzf Python-3.6.1.tar.xz
   cd Python-3.6.1
   ./configure       # sudo might be required
   make altinstall   # sudo might be required

Some **newer Linux** distributions may be able to install Python 3.6+ using your package manager.

.. code-block:: shell

   apt-get install python3.6   # sudo might be required


2) Install virtualenvwrapper
----------------------------

Install and setup virtualenvwrapper and associated tools. This allows you to create isolated "virtual environments" with independent installations of Python packages. This isn't strictly necessary, but helps prevent conflicts between incompatible package versions. To install virtualenvwrapper, run the command below:

.. code-block:: shell
   
   pip3.6 install virtualenvwrapper

An introduction and walkthrough to using the virtualenvwrapper tool is found `here <https://virtualenvwrapper.readthedocs.io/en/latest/>`__. To use virtual environments, place the following lines in the .bashrc file. The first line sets the Python interpreter for your virtual environments to python3.6. The last line is a path to your shell startup file, change it depending on where virtualenvwrapper was installed by pip.

.. code-block:: shell

   VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3.6
   export WORKON_HOME=$HOME/.virtualenvs
   export MSYS_HOME=/c/msys/1.0
   source /usr/local/bin/virtualenvwrapper.sh

.. note:: 
   
   Remember to source your .bashrc file or restart the command line.


3) Compile and Install the mcopt Library
----------------------------------------
This is the Monte Carlo code library. There are a few dependencies that must be installed before the library itself.
	
a) Install CMake
****************

`CMake <https://cmake.org/>`__ is an open-source software that controls the workflow and build process of software. To install CMake, enter the following command into the command line (using a package manager of your choice):

.. code-block:: shell

   apt-get install cmake   # sudo might be required

After installation, check the version of CMake that was installed with the following call:

.. code-block:: shell

   cmake --version

b) Install Armadillo
********************

`Armadillo <http://arma.sourceforge.net/>`__ is a wrapper that presents a clean interface to several linear algebra libraries. To install Armadillo, it is best to follow the instructions outlined `here <http://arma.sourceforge.net/download.html>`__. First, install the reccomended packages based the OS being run. Then, in the command line, use wget to download the .tar archive and extract it (use the link above to find the most recent release):

.. code-block:: shell
   
   wget http://sourceforge.net/projects/arma/files/armadillo-7.950.1.tar.xz
   tar xzf armadillo-7.950.1.tar.xz

The README.txt file found in the folder created by unpacking the Armadillo archive contains the remaining instructions. The most important of these is to change to the directory of the created folder and enter the following commands to configure armadillo:

.. code-block:: shell

   cmake .
   make
   make install   # sudo might be required

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
   ./build-unix.sh   # sudo might be required

This will place the built binary in the bin folder and run through a series of tests for correct installation.

d) Install and Compile mcopt Library
************************************

Finally, install the mcopt library itself; it can be found `here <https://github.com/jbradt/mcopt>`__. Clone the repository locally using the .git link found on GitHub.

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

   pip3.6 install Cython numpy scipy sklearn scikit-learn matplotlib seaborn jinja2 pandas clint pyYaml sqlalchemy tables h5py sphinx   # sudo might be required

Then, to install pytpc from the source code, run:

.. code-block:: shell

   python3.6 setup.py install

To test for correct installation. Run the provided tests with the following commands (not all tests print output to the screen but none should throw errors):

.. code-block:: shell
   
   cd pytpc/tests
   python3.6 -m unittest discover


6) Create a Config File
-----------------------

Create a config file for the analysis code. There is an annotated template in the next section of this documentation.


7) Set Up Energy Loss Data
--------------------------
Set up the energy loss info for the relevant nuclei.


*Tested for Ubuntu 14.04 and 16.04.*
