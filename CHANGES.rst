Changelog
~~~~~~~~~

0.6 
===
in preparation

- bootstrap the dependency on numpy
- remove custom MAD-X path discovery. You should use --madxdir if the
  library is not installed in a system location.

0.5
===
Date: 21.01.2014

- migrate to setuptools from distutils
- python3 support
- add continuous integration with Travis
- proper setup.py and MANIFEST.in to be used with PyPI
- rename package to 'cern-pymad'
- allow to build from PyPI without having cython
