Changelog
~~~~~~~~~

0.8
===
in preparation

- bootstrap the dependency on numpy
- remove custom MAD-X path discovery. You should use *--madxdir* if the
  library is not installed in a system location.
- add function ``libmadx.is_expanded``

0.7
===
Date: 16.04.2014

- close handles in remote process properly on all supported python versions
- rewrite ``libmadx.get_table`` functionality
- madx functions that return tables now return proxy objects instead. For
  backward compatibility these can be iterated to allow unpacking into a tuple
- the returned table columns is now a proxy object as well and not ``TfsTable``
- remove ``retdict`` parameter
- move some cpymad specific functionality into the cpymad package
- add libmadx/madx functions to access list of elements in a sequence


0.6
===
Date: 17.03.2014

- raise exception and don't hang up anymore, if libmadx process crashes
- on python>=3.4, close handles in remote process properly
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
