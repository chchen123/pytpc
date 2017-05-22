from setuptools import setup, Extension, Command
from setuptools.command.build_py import build_py
import numpy as np
from Cython.Build import cythonize
import sys
import os
import re


class BuildGasDBCommand(Command):
    """A command to build the gas database."""
    description = "create the gas database"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from build_gasdb import build
        build()


class BuildPyCommand(build_py):
    # Extend the build_py command to force it to build the gas DB
    def run(self):
        self.run_command('build_gasdb')
        super().run()


def get_omp_flag():
    if re.search(r'clang', os.environ.get('CC', default='')):
        return '-fopenmp=libomp'
    else:
        return '-fopenmp'


def make_extension(module, sources, language, libraries=[], extra_args=[], openmp=False):
    flags = ['-Wall', '-Wno-unused-function', '-g', '-O3']

    if language == 'c++':
        flags.append('-std=c++11')
    elif language == 'c':
        flags.append('-std=c11')

    if openmp:
        flags.append(get_omp_flag())

    if sys.platform == 'darwin':
        flags.append('-mmacosx-version-min=10.9')

    flags += extra_args

    include_dirs = [np.get_include()]

    return Extension(module, sources,
                     include_dirs=include_dirs,
                     language=language,
                     libraries=libraries,
                     extra_compile_args=flags,
                     extra_link_args=flags)


fitter_ext = make_extension(
    module='pytpc.fitting.mcopt_wrapper',
    sources=['pytpc/fitting/mcopt_wrapper.pyx'],
    language='c++',
    libraries=['mcopt'],
)

armadillo_ext = make_extension(
    module='pytpc.fitting.armadillo',
    sources=['pytpc/fitting/armadillo.pyx'],
    language='c++',
    libraries=['armadillo'],
)

cleaner_ext = make_extension(
    module='pytpc.cleaning.hough_wrapper',
    sources=['pytpc/cleaning/hough_wrapper.pyx', 'pytpc/cleaning/hough.c'],
    language='c',
    openmp=True,
)

multiplicity_ext = make_extension(
    module='pytpc.trigger.multiplicity',
    sources=['pytpc/trigger/multiplicity.pyx'],
    language='c',
)

all_extensions = [fitter_ext, armadillo_ext, cleaner_ext, multiplicity_ext]

setup(
    name='pytpc',
    version='1.1.0',
    description='Tools for analyzing AT-TPC events in Python',
    author='Joshua Bradt',
    author_email='bradt@nscl.msu.edu',
    url='https://github.com/attpc/pytpc',
    packages=[
        'pytpc',
        'pytpc.fitting',
        'pytpc.cleaning',
        'pytpc.trigger',
        'effsim',
    ],
    ext_modules=cythonize(all_extensions),
    scripts=[
        'bin/runfit',
        'bin/pyclean',
        'bin/effsim',
        'bin/convergence',
        'bin/unpack_vme',
        'bin/select_vme',
    ],
    install_requires=[
        'scipy',
        'numpy',
        'h5py',
        'tables',
        'sqlalchemy',
        'pyyaml',
    ],
    package_data={'pytpc': ['data/gases/*', 'data/raw/*', 'fitting/*.pxd']},
    extras_require={
        'docs': ['sphinx_bootstrap_theme>=0.4.5', 'sphinx>=1.2'],
        'plots': ['matplotlib', 'seaborn'],
    },
    cmdclass={
        'build_gasdb': BuildGasDBCommand,
        'build_py': BuildPyCommand,
    },
)
