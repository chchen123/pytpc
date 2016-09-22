from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
from sys import platform
from copy import deepcopy

extra_args = ['-Wall', '-Wno-unused-function', '-std=c++11', '-g']
if platform == 'darwin':
    extra_args.append('-mmacosx-version-min=10.9')

include_path = [np.get_include()]

base_kwargs = dict(include_dirs=[np.get_include()],
                   language='c++',
                   extra_compile_args=extra_args,
                   extra_link_args=extra_args)

fitter_kwargs = deepcopy(base_kwargs)
fitter_kwargs['libraries'] = ['mcopt']

cleaner_compile_args = ['-Wall', '-Wno-unused-function', '-std=c11', '-O3', '-fopenmp=libomp']
cleaner_kwargs = dict(include_dirs=[np.get_include()],
                      language='c',
                      extra_compile_args=cleaner_compile_args,
                      extra_link_args=cleaner_compile_args,
                      )


exts = [Extension('pytpc.fitting.mcopt_wrapper', ['pytpc/fitting/mcopt_wrapper.pyx'], **fitter_kwargs),
        Extension('pytpc.fitting.armadillo', ['pytpc/fitting/armadillo.pyx'], **fitter_kwargs),
        Extension('pytpc.cleaning.hough_wrapper', ['pytpc/cleaning/hough_wrapper.pyx', 'pytpc/cleaning/hough.c'],
                  **cleaner_kwargs)]

setup(
    name='pytpc',
    version='1.0.0',
    description='Tools for analyzing AT-TPC events in Python',
    author='Joshua Bradt',
    author_email='bradt@nscl.msu.edu',
    url='https://github.com/attpc/pytpc',
    packages=['pytpc', 'pytpc.fitting', 'pytpc.cleaning'],
    ext_modules=cythonize(exts),
    scripts=['bin/runfit', 'bin/pyclean'],
    install_requires=['scipy',
                      'numpy',
                      'h5py',
                      'tables'],
    package_data={'pytpc': ['data/gases/*', 'data/raw/*', 'fitting/*.pxd']},
    extras_require={
        'docs': ['sphinx_bootstrap_theme>=0.4.5', 'sphinx>=1.2'],
        'plots': ['matplotlib', 'seaborn'],
    },
)
