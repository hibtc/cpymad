Changelog
~~~~~~~~~

0.7
===
in preparation

- bootstrap the dependency on numpy
- remove custom MAD-X path discovery. You should use --madxdir if the
  library is not installed in a system location.

0.6
===
Date: 17.03.2014

- raise exception and don't hang up anymore, if libmadx process crashes
- on python>=2.7, close handles in remote process properly
- let every 'Madx' instance have an independent copy of the madx library.
  this makes the madx module much more useful. previously, this was only
  true for instances of 'cpymad.model'.
- restrict to only one cython module that links to libmadx. (allows static
  linking which is advantageous on windows!)
- use YAML model files instead of JSON
- make 'madx' a submodule of 'cpymad'
- fix test exit status

0.5
===
Date: 21.01.2014

- migrate to setuptools from distutils
- python3 support
- add continuous integration with Travis
- proper setup.py and MANIFEST.in to be used with PyPI
- rename package to 'cern-pymad'
- allow to build from PyPI without having cython
