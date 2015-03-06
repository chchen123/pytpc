from setuptools import setup, find_packages

setup(name='pytpc',
      version='0.1',
      description='Tools for analyzing TPC events in Python',
      author='Joshua Bradt',
      author_email='bradt@nscl.msu.edu',
      packages=['pytpc'],
      install_requires=['sphinx_bootstrap_theme>=0.4.5',
                        'sphinx>=1.2',
                        'scikit-learn',
                        'seaborn',
                        'matplotlib',
                        'scipy',
                        'numpy'],
      )