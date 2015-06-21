Changelog
~~~~~~~~~

(dates are in the form dd.mm.yyyy)


0.10.7
======
Date: 21.06.2015

- allow redirection of MAD-X standard I/O via Madx constructor


0.10.6
======
Date: 29.05.2015

- add csv() method for ResourceProvider
- use C loader from yaml for performance if available
- convert madx.metadata.get_copyright_notice
- add accessors to real sequence + elements for model.Sequence


0.10.5
======
Date: 25.05.2015

- add MAD-X specific metadata in cpymad.madx.metadata
- speedup Travis testing (using caches and docker containers)


0.10.4
======
Date: 22.04.2015

- prevent MAD-X process from exiting on Ctrl-C (this was an especially
  nasty feature when using the interactive python interpretor)
- upgrade to `MAD-X 5.02.05`_ (development release from 10.04.2015)
- fix leakage of open file handles into remote process on py2/windows

.. _`MAD-X 5.02.05`: http://madx.web.cern.ch/madx/releases/5.02.05/


0.10.3
======
Date: 29.03.2015

- make sequence.elements.index more convenient: can now handle names with or
  without the ':d' suffix as well as the special names '#s' and '#e'


0.10.2
======
Date: 05.03.2015

- add some utility functions to work with MAD-X element names and identifiers
- add a setter method for values to Madx
- improve install instructions. In particular, recommend WinPython as build
  environment
- fix the MinGW build error due to broken sysconfig inline
- run setup only if invoked as main script


0.10.1
======
Date: 09.01.2015

- convert IOError to RemoteProcessCrashed, which can occur on transmission
  if the remote process is already down
- convert ValueError to RemoteProcessClosed, which can occur on transmission
  if the remote process was already closed


0.10.0 Fork
===========
Date: 09.01.2015

This is the first independent version released for the `HIT cpymad fork`_.
The changes on the public API are so heavy, that this is basically a new
library.

- rename package from ``cern.cpymad`` to ``cpymad``
- remove LHC models from repository
- redesign API to make more use of OOP (no stable API yet!)
- removed some obsolete / unused modules

.. _HIT cpymad fork: https://github.com/hibtc/cpymad


0.9
===
Date: 17.11.2014

- don't link against numpy anymore (this makes distribution of prebuilt
  binaries on windows actually useful)
- add MAD-X license notice (required to distribute binaries)
- setup.py doesn't require setuptools to be pre-installed anymore (if
  internet is available)
- some doc-fixes
- convert cpymad._couch to a simple module (was a single file package)
- use ``logging`` through-out the project
- alow logger to be specified as model/madx constructor argument
- multi-column access, e.g.: ``table.columns['betx','bety']``
- move tests one folder level up


0.8
===
Date: 30.06.2014

- isolate cpymad: remove jpymad backend, remove pymad base
- bootstrap the dependency on numpy
- remove custom MAD-X path discovery during setup. You should use
  *--madxdir* if the library is not installed in a system location.
- add function ``libmadx.is_expanded``
- add function ``libmadx.chdir``
- handle MAD-X table columns with integer arrays
- make ``madx.command`` more powerful (allows ``**kwargs`` and attribute
  access)
- use inherited pipes for IPC with remote MAD-X processes (allows to
  forward stdin/stdout separately)
- close connection to remote process on finalization of ``LibMadxClient``
- remove MAD-X command checks, ``recursive_history`` and filename
  completion
- fix name clash
- fix some bugs
- rename convenience constructors to ``cern.cpymad.load_model`` and
  ``cern.cpymad.start_madx`` due to name clash with module names


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
