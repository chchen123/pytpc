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

The code itself was written and tested with Python 3.3, but it *should* also work with Python 2.6+. 

## Documentation

Read the documentation online at: https://groups.nscl.msu.edu/attpc/doc/pytpc

Most of the code has documentation written into it in the form of docstrings. There is also Sphinx documentation, which
can be built by running `make html` from inside the `docs` directory.

## Contact

Josh Bradt, bradt@nscl.msu.edu