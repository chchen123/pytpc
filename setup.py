from setuptools import setup

setup(
    name='pytpc',
    version='0.7.1',
    description='Tools for analyzing TPC events in Python',
    author='Joshua Bradt',
    author_email='bradt@nscl.msu.edu',
    url='https://github.com/attpc/pytpc',
    packages=['pytpc'],
    install_requires=['scikit-learn',
                      'seaborn',
                      'matplotlib',
                      'scipy',
                      'numpy',
                      'h5py'],
    package_data={'pytpc': ['data/gases/*', 'data/raw/*']},

    extras_require={
        'docs': ['sphinx_bootstrap_theme>=0.4.5', 'sphinx>=1.2'],
    },
    )
