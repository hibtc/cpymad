Changelog
~~~~~~~~~

1.4.0
=====
Date: 06.07.2019

- expose sequence length as ``Sequence.length``


1.3.0
=====
Date: 22.06.2019

- expose MAD-X errors as ``element.field_errors``, ``.phase_errors``, and
  ``.align_errors`` when accessed through sequence


1.2.2
=====
Date: 12.06.2019

- update to MAD-X 5.05.01


1.2.1
=====
Date: 05.06.2019

- fix deadlock if accessing the global ``cpymad.madx.metadata`` object with
  closed or invalid STDIN (after ``os.close(0)`` or in windows GUI application)


1.2.0
=====
Date: 11.05.2019

- update to MAD-X 5.05.00
- implement Madx.chdir using the new CHDIR command. This improves readability
  and repeatability of command histories.


1.1.2
=====
Date: 13.04.2019

- expose all columns in table, don't limit by current selection
- unify the get_table_column_XXX functions in libmadx
- add Table.selected_columns method
- add Table.selected_rows method
- drop ability to build MAD-X through setup.py
- don't search for MAD-X in system locations
- simplifications in setup script
- replace runtime dependency on setuptools by importlib_resources


1.1.1
=====
Date: 18.02.2019

- build with GC 8.0.2 on windows
- build 32bit wheels for linux


1.1.0
=====
Date: 16.02.2019

- add ``AttrDict.update()`` method similar to regular dicts
- add ``Table.row_names()`` query method
- use row names as table index for pandas dataframe
- add ``Madx.batch()`` context manager to collect commands before sending them
  to MAD-X in a single batch all at once (performance)
- add a convenience parameter ``Madx(history=[])`` to simplify capturing
  history
- explicitly specify ``zip_safe=False`` for the cpymad package. This will work
  better for builds against shared MAD-X library
- close CommandLog files when calling ``Madx.quit()`` (if they were opened
  by us)

Finally, this is the first release to automate the release process for windows
wheels:

- build windows wheels on appveyor
- upload windows wheels to pypi on tags
- test cpymad on windows using appveyor


1.0.11
======
Date: 18.01.2019

- guard ``expr_vars`` against passing ``None`` etc
- add ``elems`` parameter to ``normalize_range_name``


1.0.10.post1
============
Date: 11.12.2018

- build windows wheels with bdwgc 7.6.8 to mitigate problems on win10


1.0.10
======
Date: 07.12.2018

- fix broken caching logic in travis config
- move type constants to ``cpymad.types``
- export a MAD-X dtype to python type mapping from ``cpymad.types``


1.0.9
=====
Date: 21.11.2018

- suppress internal stack traces
- raise exception for failed twiss instead of returning invalid table that
  will crash later on
- fix incorrect ``Element.position`` attribute for sequences with
  ``refer=entry`` or ``refer=exit``
- allow passing parameters with underscore suffix to commands, this allows
  passing parameters as bare words that conflict with python keywords (e.g.
  ``madx.command.select(class_='quadrupole')``
- improve ``repr()`` for ``Table``: show column names


1.0.8
=====
Date: 18.10.2018

- add ``Table.dframe()`` method to return pandas dataframe (provisional API)
- return success status from ``Madx.input`` (MAD-X errorflag)
- update install instructions to account for symbol visibility
- fix MAD-X crash on errors due to interposition of ``error`` by libc (linux)
- automatically update documentation from travis
- add ``quit`` method to shutdown the interpreter and wait for the process
- fix file deletion in case of errors within ``temp_filename`` context
- make ``Madx`` usable as context manager
- use the correct line continuation in .bat example
- fix manylinux build error: not creating libmadx.c
- fix "Permission denied" error when having to clone MAD-X etc
- fix incorrect ABI in the -cp27mu- wheel
- allow specifying MAD-X/cpymad source tarballs for manylinux build
- rework usage of data volumes in manylinux container: readonly cpymad folder
- automatically build and upload manylinux releases from travis!
- fix rare race-condition in stream reader -> delayed output
- replace some magic numbers with human readable names
- use ``coverage combine`` instead of custom path hack before uploading
  coverage data
- move cpymad package to unimportable subdirectory
- update to MAD-X 5.04.02


1.0.7
=====
Date: 19.09.2018

- fix io.UnsupportedOperation on python2 when sys.stdout is not a file
- increase minrpc dependency to better comply with redirected stdouts
- fix DeprecationWarning due to not importing ABCs from collections.abc
- improvements in test suite and automatic style checks


1.0.6
=====
Date: 28.08.2018

- remove unused ``error_log`` from ``Madx``
- support passing arbitrary callables to ``Madx(stdout=...)``
- support passing non-file ``IOBase`` objects as ``stdout``
- default to ``sys.stdout``
- can pass almost arbitrary MAD-X scripts to ``input``, including
  comments/multiline commands


1.0.5
=====
Date: 16.08.2018

API:
- rename ``Madx.call``'s first parameter as in MAD-X
- handle all ``USE`` parameters in ``Madx.use``

setup:
- provide manylinux wheels!
- remove spurious dependency on pyyaml
- finally get the real meaning of MADX_STATIC and BUILD_SHARED_LIBS
- default to BUILD_SHARED_LIBS=OFF on all platforms
- rework arguments for setup.py
- make linking against X11 optional (requires unreleased MAD-X ``5.04.02``)
- default to X11=OFF if building MAD-X
- improve documentation


1.0.4.post1
===========
Date: 24.07.2018

- fix py2 syntax error in setup.py


1.0.4
=====
Date: 25.07.2018

- add ``limits`` parameter to ``Madx.match``
- try to download and build MAD-X in setup.py if it is not already available


1.0.3.post1
===========
Date: 15.07.2018

- Build windows wheels against the real (July) 5.04.01 release of MAD-X


1.0.3
=====
Date: 02.07.2018

- Fix ValueError for missing values in ElementList.get


1.0.2
=====
Date: 25.06.2018

Increase test coverage up to ``96%`` (from 75), and fix a few minor bugs
detected in the tests:

Command composition:
- handle composite ranges (``A/B``) in
- fix AttributeError when composing command with equality ``Constraint``
- fix incorrect output for STRING_ARRAY range parameters (MATCH)
- fix passing ``Range`` objects as ranges

Misc:
- fix table column names being ``bytes``, return as unicode ``str``
- fix the ``sectortable2`` method
- add ``Madx.options`` property that allows to view the current set of options
- remove unused helper method ``Sequence._parse_range``
- return the cloned element from ``Element.clone``


1.0.1
=====
Date: 22.06.2018

- improve error message on missing command attributes
- allow negative indices when accessing table rows
- fix returning the correct table from ``twiss()``/``survey()`` if a
  non-default table is used
- improve support for multi-line commands in ``input()`` (but still no comments!)
- automatically add missing semicolons at the end of command strings


1.0.0
=====
Date: 11.06.2018

Please see the comprehensive list of changes and backward incompatibilities
mentioned in the prereleases!

In addition:

- update to minrpc 0.0.7
- the windows wheels are built using MAD-X 5.04.01


1.0.0rc3
========
Date: 31.05.2018

- add ``Parameter.var_type`` that tells apart constant/direct/expression vars
- change the meaning of ``inform`` for globals, ``inform=0`` means now
  "predefined variable"
- fix TypeError occuring in ``mad_command`` when composing string arrays


1.0.0rc2
========
Date: 15.05.2018

- serve globals as ``Parameter`` instances from libmadx module
- add ``cmdpar`` attribute to ``globals``


1.0.0rc1
========
Date: 13.05.2018

Collecting further backward incompatibilities before the final 1.0 release, as
well as minor bugfixes.

- only execute variable updates if their value has changed
- use ``__slots__`` for Parameter
- remove cpymad-specific behaviour for ``Madx.select``
- allow direct access to MAD-X commands as attributes on the ``Madx`` instance
- rename ``Parameter.argument`` to ``Parameter.definition``
- add ``VarList.defs``, ``Command.defs`` instance variables for accessing the
  definitions (provisional API)
- create the accessor proxies in advance
- disallow indexing ``ElementList`` by ``dict`` instances (i.e. by element
  object)
- handle uppercase '#S' and '#E' in ``ElementList``


1.0.0rc0
========
Date: 16.04.2018

First pre-release for 1.0.0 with several backward incompatibilities.

- the Madx methods have been simplified to be only thin wrappers over the
  corresponding MAD-X commands, not taking any extra responsibilities such as
  automatically using sequences etc.
- rename ``Madx.tables/sequences`` to singular form
- disable passing dicts as range parameter for commands
- remove ``cpymad.util.is_match_param``.
- remove ``cpymad.libmadx.set_var`` routine. Always use ``input``!
- remove ``Madx.get_table`` method, use ``Madx.table.X`` instead
- rename ``Madx.evaluate`` to ``eval``
- remove ``Madx.set_value/set_expression/update_value``. Use assignment to
  attributes of ``Madx.globals/command/element`` instead.
- rename ``util.mad_command`` -> ``format_command``
- only ignore ``None`` parameters when generating MAD-X commands. This allows
  passing empty strings.
- remove ``cpymad.types.Expression``, replaced by new ``Parameter`` class, see
  below.
- remove ``Madx.active_sequence``, use ``Madx.sequence()`` instead
- the ``at/l`` attributes are now kept as the values specified by the user
  (relative to *refer* flag), and not overwritten anymore by the actual
  position or length. Use ``.position`` and ``.length`` attributes to access
  the node position/length instead!
- the ``name`` attribute is now the command/element name. The node name is
  now available as ``node_name``.

Introduced a new API for accessing additional metadata about command
parameters:

- added a ``Command.cmdpar.X`` namespace that can be used to retrieve a
  ``Parameter`` instance with additional metadata about the command parameter.
- rigorously distinguish between MAD-X command parameters and other attributes
  on elements/commands
- only command parameters can be accessed using the dict-like item access
  syntax while other metadata can only be accessed via attribute access
- use the type information for improving the composition of MAD-X command
  statements

Misc changes:

- add method ``Madx.sectortable2`` to access 2nd order sector map (as well as
  related methods to ``Table``). Method name is subject to change!
- show implicit drifts with ``[0]`` again (the suffix is needed when matching
  on implicit drifts)
- perfect kwargs forwarding
- expose ``occ_count/enable/base_name`` attributes on nodes


0.19.1
======
Date: 02.04.2018

- pass unescaped (raw) string arguments to MAD-X
- use double-quotes by default
- overload ``Madx.evaluate`` for floats and lists (making it applicable
  for anything that may be returned in the property)
- windows builds link against MAD-X 49b4e7fee "Fix incorrect field errors
  in tmbend with INTERPOLATE". This is a few minor bugfixes after 5.04.00.


0.19.0
======
Date: 25.03.2018

- command/element etc:
    * retrieve information about commands from MAD-X ``defined_commands`` and
      store in ``Command`` instances.
    * use ``Command`` to improve command string generation and type-checks in
      ``util.mad_command`` (#9)
    * quote filename parameters when composing command string
    * use deferred expressions (``:=``) whenever passing strings to
      non-string parameters (#11)
    * subclass elements, beam from ``Command``
    * support attribute access for table/mappings/commands/elements/beams etc
    * allow case-insensitive access
    * overload index-access in tables to retrieve rows
    * implement ``Element.__delitem__`` by setting value to default
    * return name for global elements too
    * add ``Madx.base_types`` data variable that yields the base elements
    * add ``Element.parent``/``base_type`` attributes
    * more concise string representations
    * strip -Proxy suffix from class names
    * apply user defined row/column selections even when no output file is
      specified

- installation:
    * automatically use ``-lquadmath``
    * add ``--static`` flag for setup script, use ``--shared`` by default
    * no more need to link against PTC shared object separately
    * finally provide some binary wheels for py 3.5 and 3.6 (#32)

- raise cython language_level to 3
- require MAD-X 5.04.00


0.18.2
======
Date: 05.12.2017

- fix order of ``weight`` command in ``Madx.match``


0.18.1
======
Date: 30.11.2017

- fix some inconsistencies regarding the mixture of unicode and byte strings
  on python2 (NOTE: still expected to be broken!)
- provide copyright notice as unicode


0.18.0
======
Date: 16.11.2017

- if no table columns are selected, show all by default
- need setuptools>=18.0
- thread-support:
    - release GIL during ``input()``
    - can specify a lock for minrpc


0.17.4
======
Date: 24.10.2017

- replace Madx.get_transfer_map_7d method
- require ``MAD-X 5.03.07`` (bugfix n_nodes)
- documentation improvements


0.17.3
======
Date: 02.07.2017

- TableProxy gets getmat method for retrieving (sigma/r) matrices
- update official support to ``MAD-X 5.03.06``


0.17.2
======
Date: 29.05.2017

- keep user specified argument order (only py>=3.6)
- update official support to ``MAD-X 5.03.05``
- build the windows version with GC


0.17.1
======
Date: 22.05.2017

- fix ``Madx.help(topic)``
- fix string decoding for namelists on python3
- improve interactive display of proxy objects
- improve default flags for starting the libmadx subprocess


0.17.0
======
Date: 16.02.2017

- update official support to ``MAD-X 5.02.13``
- cache columns in TableProxy
- add fast functions to obtain element positions


0.16.0
======
Date: 06.12.2016

- add efficient functions to get list of all element namems
- provide element index within the sequence
- add function to check MAD-X expressions
- make ``evaluate()`` slightly safer
- add windows build scripts (``.bat``)


0.15.2
======
Date: 16.10.2016

- Update official support to ``MAD-X 5.02.12``


0.15.1
======
Date: 13.10.2016

- Update official support to ``MAD-X 5.02.11``


0.15.0
======
Date: 24.09.2016

- depend on *minrpc* for RPC
- in setup: fix ``NameError: force_lib`` on Mac


0.14.3
======
Date: 15.09.2016

- in setup: disable unsupported ``--no-as-needed`` on Mac
- in setup: allow value of ``--madxdir`` to be specified as separate argument
- format ``types.Expression`` in commands
- fix formatting of ``types.Constraint``


0.14.2
======
Date: 12.09.2016

- don't ignore exceptions from ``clibmadx._get_node_index``
- fix node positions for auto-inserted DRIFTs
- fix node positions for unexpanded sequences
- add some methods for sequence expansion
- change ``libmadx.get_table_column_count()`` to return number of *selected*
  columns for consistency
- fix bug in ``Madx._use()`` that caused ``Madx.twiss()`` and other functions
  to reUSE the sequence and thus clear previously selected flags
- force linking against libptc (required on py35 if MAD-X is installed in
  non-system location, since ``DT_RUNPATH`` is non-transitive and does not
  resolve libptc as indirect dependency via the chain cpymad->libmadx->libptc)


0.14.1
======
Date: 18.05.2016

- improve MAD-X command composition
- Update official support to ``MAD-X 5.02.10``


0.14.0
======
Date: 04.03.2016

- Add function to obtain transfer map
- Fix bug with expanded_elements listing too many elements (leading elements
  were re-listed at the end)


0.13.0
======
Date: 24.01.2016

- Update to ``MAD-X 5.02.08``:
    - official support
    - automatic tests
    - prebuilt binaries on PyPI (for windows)


0.12.2
======
Date: 30.10.2015

- Strip trailing underscore from ``MadxCommands`` attribute names. This allows
  the syntax to be used even for python keywords, e.g. ``m.command.global_()``
- Change the behaviour of ``Madx.globals``:
    - when setting string values, set the variable as deferred expression
    - when getting deferred variables, return instances of type ``Expression``
    - when iterating, only show non-constant globals


0.12.1
======
Date: 13.10.2015

- fix crash due to incorrect parameter name for ``logging.basicConfig``
- fix crash due to missing ``subprocess.MAXFD`` on python3.5
- fix coverage report submitted to coveralls.io


0.12.0
======
Date: 05.10.2015

- expose directory of global MAD-X variables as ``Madx.globals``
- expose directory of global MAD-X elements as ``Madx.elements``
- fix a bug with Elements.__contains__ reporting yes incorrectly
- list only those column of a table that are marked for output
- add function to get row names of a table


0.11.0
======
Date: 03.07.2015

- Remove models + resource handling from cpymad. If you need these, check
  them out from the previous version and maintain them in your own code
  base. This way you are much more flexible to adapt models to your needs.


0.10.8
======
Date: 02.07.2015

- Public element names are now of the form "foo[3]" or simply "foo". The
  syntax "foo:d" can not be used anymore (this form is used by MAD-X only
  internally and converted at the lowest wrapper level).
- Fix exception when not specifying sequence name on Madx methods


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
