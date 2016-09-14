from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
from sys import platform

extra_args = ['-Wall', '-Wno-unused-function', '-std=c++11', '-g']
if platform == 'darwin':
    extra_args.append('-mmacosx-version-min=10.9')

include_path = [np.get_include()]

ext_kwargs = dict(include_dirs=[np.get_include()],
                  libraries=['mcopt'],
                  language='c++',
                  extra_compile_args=extra_args,
                  extra_link_args=extra_args)

exts = [Extension('atmc.mcopt_wrapper', ['atmc/mcopt_wrapper.pyx'], **ext_kwargs),
        Extension('atmc.armadillo', ['atmc/armadillo.pyx'], **ext_kwargs)]

setup(
    name='pytpc',
    version='1.0.0',
    description='Tools for analyzing TPC events in Python',
    author='Joshua Bradt',
    author_email='bradt@nscl.msu.edu',
    url='https://github.com/attpc/pytpc',
    packages=['pytpc'],
    install_requires=['scipy',
                      'numpy',
                      'h5py',
                      'tables'],
    package_data={'pytpc': ['data/gases/*', 'data/raw/*']},
    extras_require={
        'docs': ['sphinx_bootstrap_theme>=0.4.5', 'sphinx>=1.2'],
        'plots': ['matplotlib', 'seaborn'],
    },
    )

# setup(name='atmc',
#       version='2.1.0',
#       description='Particle tracking and MC optimizer module',
#       packages=['atmc'],
#       package_data={'atmc': ['*.pxd']},
#       ext_modules=cythonize(exts),
#       )
