from distutils.core import setup

setup(name='pytpc',
      version='0.1',
      description='Tools for analyzing TPC events in Python',
      author='Joshua Bradt',
      author_email='bradt@nscl.msu.edu',
      modules=['pytpc/tracking.py', 'pytpc/simulation.py', 'pytpc/constants.py',
               'pytpc/evtdata.py', 'pytpc/kalman.py', 'pytpc/tpcplot.py'],
      requires=['numpy', 'matplotlib', 'filterpy', 'scikit-learn', 'scipy']
      )