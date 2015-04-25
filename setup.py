from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(name='pytpc',
      version='0.1',
      description='Tools for analyzing TPC events in Python',
      author='Joshua Bradt',
      author_email='bradt@nscl.msu.edu',
      packages=['pytpc'],
      ext_modules=cythonize('pytpc/simulation.pyx', annotate=True),
      install_requires=['sphinx_bootstrap_theme>=0.4.5',
                        'sphinx>=1.2',
                        'scikit-learn',
                        'seaborn',
                        'matplotlib',
                        'scipy',
                        'numpy'],
      include_dirs=[np.get_include()]
      )