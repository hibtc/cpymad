cpymad
------
|Version| |Downloads| |License| |Python|

cpymad is a Cython_ binding to MAD-X_.

MAD-X is a software package to simulate particle accelerators and is used
at CERN and all around the world. It has its own proprietary scripting
language and is usually launched from the command line.

This version of cpymad is tested with MAD-X |VERSION|. Other MAD-X
versions (and immediate revisions) might work too, but are more likely to
cause problems.

The installable wheel archives that are provided for some versions of
python to simplify the installation on windows contain a precompiled
version of cpymad that is statically linked against MAD-X |VERSION|.

.. _Cython: http://cython.org/
.. _MAD-X: http://cern.ch/mad
.. |VERSION| replace:: 5.02.12


Disclaimer
~~~~~~~~~~

This is a heavily modified fork of the cern-cpymad_ package. The fork is
not authored or maintained by CERN members.

cpymad links against an unofficial build of MAD-X that is not supported by
CERN, i.e. this binary may have problems that the official binary does not
have and vice versa. This means:

- Only report issues to CERN that can be reproduced with their official
  command line client.
- Only report issues here if they cannot be reproduced with their official
  command line client.

See `Reporting issues`_.

.. _cern-cpymad: https://github.com/pymad/cpymad


Project pages
~~~~~~~~~~~~~

- `Installation`_
- `Source code`_
- `Documentation`_
- `Issue tracker`_
- `Releases`_

.. _Installation: http://hibtc.github.io/cpymad/installation
.. _Source code: https://github.com/hibtc/cpymad
.. _Documentation: http://hibtc.github.io/cpymad
.. _Issue tracker: https://github.com/hibtc/cpymad/issues
.. _Releases: https://pypi.python.org/pypi/cpymad


Usage
~~~~~

The ``Madx`` class provides a basic binding to the MAD-X interpreter:

.. code-block:: python

    from cpymad.madx import Madx

    # Start a MAD-X interpretor. All MAD-X commands issued via cpymad will
    # be logged to `command_log`:
    madx = Madx(command_log="log.madx")

    # Show the version of MAD-X that is actually loaded:
    print(madx.version)

    # Execute one of your predefined MAD-X files:
    madx.call('/path/to/some/input_file.madx')

    # Only a handful of MAD-X methods are exposed as methods. For others,
    # you can use the `command` attribute. For example, to set a beam:
    madx.command.beam(sequence='myseq1', particle='PROTON')

    # Calculate TWISS parameters:
    twiss = madx.twiss(sequence='LEBT',
                       betx=0.1, bety=0.1,
                       alfx=0.1, alfy=0.1)

    # Your own analysis below:
    from matplotlib import pyplot as plt
    plt.plot(twiss['s'], twiss['betx'])
    plt.show()

There are alternative syntaxes for the extreme cases where you need more
fine grained control over the command string composition or where
``command`` fails to do the right thing:

.. code-block:: python

    # can't use `global` as attribute, since it's a python keyword:
    madx.command.global_(sequence='cassps', Q1=26.58)

    # can't use `: ` as attribute:
    madx.command('QP: QUADRUPOLE', AT=2, L=1)

    # issue a plain text command, don't forget the semicolon!
    madx.input('FOO, BAR=[baz], QUX=<NORF>;')


See http://hibtc.github.io/cpymad for further documentation.


Known issues
~~~~~~~~~~~~

On windows with python3.3, there is currently no satisfying way to close file
handles in the MAD-X process or prevent them from being inherited by default.
You have to make sure on your own that you close all file handles before
creating a new ``cpymad.madx.Madx`` instance!


Hacking
~~~~~~~

Try to be consistent with the PEP8_ guidelines. Add `unit tests`_ for all
non-trivial functionality. `Dependency injection`_ is a great pattern to
keep modules testable.

Commits should be reversible, independent units if possible. Use descriptive
titles and also add an explaining commit message unless the modification is
trivial. See also: `A Note About Git Commit Messages`_.

.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _`unit tests`: http://docs.python.org/2/library/unittest.html
.. _`Dependency injection`: http://www.youtube.com/watch?v=RlfLCWKxHJ0
.. _`A Note About Git Commit Messages`: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html


Reporting issues
~~~~~~~~~~~~~~~~

If you have a problem with a sequence file, first try to check if that
problem remains when using the MAD-X command line client distributed by
CERN, then:

- Report the issue to CERN only if it be reproduced with their official
  command line client.
- Report the issue here only if it cannot be reproduced with their official
  command line client.

For issues regarding the cpymad code itself or usage information, I'm happy to
answer. Just keep in mind to be **precise**, **specific**, **concise** and
provide all the necessary information.

See also:

- `Short, Self Contained, Correct (Compilable), Example`_
- `How to Report Bugs Effectively`_
- `How To Ask Questions The Smart Way`_.

.. _Short, Self Contained, Correct (Compilable), Example: http://sscce.org/
.. _How to Report Bugs Effectively: http://www.chiark.greenend.org.uk/~sgtatham/bugs.html
.. _How To Ask Questions The Smart Way: http://www.catb.org/esr/faqs/smart-questions.html


Tests
~~~~~

Currently, tests run on:

- The `Travis CI`_ service is mainly used to check that the unit tests for
  pymad itself execute on several python versions. Python{2.7,3.3} are
  supported. The tests are executed on any update of an upstream branch.
  The Travis builds use a unofficial precompiled libmadx-dev_ package to
  avoid having to rebuild the entire MAD-X library on each invocation.

  |Build| |Coverage|

.. _`Travis CI`: https://travis-ci.org/hibtc/cpymad
.. _libmadx-dev: https://github.com/hibtc/madx-debian


.. |Build| image:: https://api.travis-ci.org/hibtc/cpymad.svg?branch=master
   :target: https://travis-ci.org/hibtc/cpymad
   :alt: Build Status

.. |Coverage| image:: https://coveralls.io/repos/hibtc/cpymad/badge.svg?branch=master
   :target: https://coveralls.io/r/hibtc/cpymad
   :alt: Coverage

.. |Version| image:: http://coldfix.de:8080/v/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad/
   :alt: Latest Version

.. |Downloads| image:: http://coldfix.de:8080/d/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad#downloads
   :alt: Downloads

.. |License| image:: http://img.shields.io/badge/license-CC0,_Apache,_Non--Free-red.svg
   :target: https://github.com/hibtc/cpymad/blob/master/COPYING.rst
   :alt: License

.. |Python| image:: http://coldfix.de:8080/py_versions/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad#downloads
   :alt: Supported Python versions
