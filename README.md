# pytpc

This package provides a Python framework for analyzing data from the Active-Target Time Projection Chamber (AT-TPC) at
the NSCL.

## Dependencies

The following packages are required to use pytpc:

- numpy
- matplotlib
- scipy
- scikit-learn
- seaborn
- sphinx and sphinx-bootstrap-theme (if you want to build the documentation)

These should be installed automatically if you use the setup.py script.

The code itself was written and tested with Python 3.4. I don't believe it will work with Python 2.7, but anything
is possible.

## Recommended installation procedure

The easiest way to install the code is by satisfying the dependencies with Conda. Download and install `conda` from
http://continuum.io/downloads (make sure you get the Python 3 version). Then install the dependencies with

```bash
conda install numpy scipy scikit-learn matplotlib seaborn
```

Next, if you're installing pytpc from the source code, run
```bash
python setup.py install
```

## Documentation

Read the documentation online at: https://groups.nscl.msu.edu/attpc/doc/pytpc

Most of the code has documentation written into it in the form of docstrings. There is also Sphinx documentation, which
can be built by running `make html` from inside the `docs` directory.

## Contact

Josh Bradt, bradt@nscl.msu.edu
