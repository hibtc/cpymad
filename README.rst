CPyMAD
------
|Version| |Downloads| |License| |Python|

CPyMAD_ is a Cython_ binding to MAD-X_.

MAD-X is a software package to simulate particle accelerators and is used
at CERN and all around the world. It has its own proprietary scripting
language and is usually launched from the command line.

.. _CPyMAD: https://github.com/hibtc/cpymad
.. _Cython: http://cython.org/
.. _MAD-X: http://cern.ch/mad
.. |VERSION| replace:: 5.02.05

This version of CPyMAD is tested with MAD-X |VERSION|. Other MAD-X
versions (and immediate revisions) might work too, but are more likely to
cause problems.

The installable wheel archives that are provided for some versions of
python to simplify the installation on windows contain a precompiled
version of CPyMAD that is statically linked against MAD-X |VERSION|.


Disclaimer
~~~~~~~~~~

This is a heavily modified fork of the cern-cpymad_ package. The fork is
not authored or maintained by CERN members.

CPyMAD links against an unofficial build of MAD-X that is not supported by
CERN, i.e. in case of problems you will not get help there.

.. _cern-cpymad: https://github.com/pymad/cpymad


Installation
~~~~~~~~~~~~

See http://hibtc.github.io/cpymad/installation.


Usage
~~~~~

The ``Madx`` class provides a basic binding to the MAD-X interpreter:

.. code-block:: python

    from cpymad.madx import Madx

    # create a new interpreter instance:
    # the optional 'command_log' parameter can be used to store MAD-X
    # command history.
    madx = Madx(command_log="log.madx")

    # determine the version of MAD-X that is actually loaded:
    print(madx.version)

    # you execute arbitrary textual MAD-X commands:
    madx.input('call, file="input_file.madx";')

    # there is a more convenient syntax available which does the same:
    madx.command.call(file="input_file.madx")

    # And for some commands there exist direct shortcuts:
    madx.call('/path/to/some/input_file.madx')

    # Calculate TWISS parameters:
    twiss = madx.twiss(sequence='LEBT',
                       betx=0.1, bety=0.1,
                       alfx=0.1, alfy=0.1)

    # Your own analysis below:
    from matplotlib import pyplot as plt
    plt.plot(twiss['s'], twiss['betx'])
    plt.show()

There is also a ``Model`` class which encapsulates more metadata for complex
accelerator machines. If you have ready-to-use model definitions on your
filesystem, models can be instanciated and used as follows:

.. code-block:: python

    from cpymad.model import Model

    model = Model.load('/path/to/model/definition.cpymad.yml')

    for sequence in model.sequences.values():
        twiss = sequence.twiss()

See http://hibtc.github.io/cpymad for further documentation.


Contributing
~~~~~~~~~~~~

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

.. |Version| image:: https://pypip.in/v/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad/
   :alt: Latest Version

.. |Downloads| image:: https://pypip.in/d/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad#downloads
   :alt: Downloads

.. |License| image:: http://img.shields.io/badge/license-CC0,_Apache,_Non--Free-red.svg
   :target: https://github.com/hibtc/cpymad/blob/master/COPYING.rst
   :alt: License

.. |Python| image:: https://pypip.in/py_versions/cpymad/badge.svg
   :target: https://pypi.python.org/pypi/cpymad#downloads
   :alt: Supported Python versions
