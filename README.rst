CPyMAD
------

CPyMad_ is a Cython_ binding to MAD-X_.

MAD-X is a software package to simulate particle accelerators and is used
at CERN and all around the world. It has its own proprietary scripting
language and is usually launched from the command line.

There is also a binding via JMad_ called JPyMAD_. This has less features
but does not require a C compiler for installing.

.. _CPyMAD: https://github.com/pymad/cpymad
.. _Cython: http://cython.org/
.. _MAD-X: http://cern.ch/mad
.. _JMad: http://jmad.web.cern.ch/jmad/
.. _JPyMAD: https://github.com/pymad/jpymad

**IMPORTANT:** cern-cpymad links against an unofficial build of MAD-X that
is not supported by CERN, i.e. in case of problems you will not get help
there.


Dependencies
------------

To build MAD-X and CPyMAD from source you will need

- CMake_
- python>=2.6
- C / Fortran compilers.

Furthermore, CPyMAD depends on the following python packages:

- setuptools_
- Cython_
- NumPy_
- PyYAML_

The python packages can be installed using pip_.


.. _CMake: http://www.cmake.org/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _Cython: http://cython.org/
.. _NumPy: http://www.numpy.org/
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
.. _pip: https://pypi.python.org/pypi/pip


Installation
~~~~~~~~~~~~

Installation instructions are available at http://pymad.github.io/cpymad/installation.


Usage
~~~~~

.. code-block:: python

    from cern import cpymad

    # Instanciate a model:
    m = cpymad.load_model('lhc')

    # Calculate TWISS parameters:
    twiss, summary = m.twiss()

    # Your own analysis below:
    from matplotlib import pyplot as plt
    plt.plot(twiss['s'], twiss['betx'])


See http://pymad.github.io/cpymad for further documentation.


Development guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Coding:**

Try to be consistent with the PEP8_ guidelines as far as you are familiar
with it. Add `unit tests`_ for all non-trivial functionality.
`Dependency injection`_ is a great pattern to keep modules testable.

.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _`unit tests`: http://docs.python.org/2/library/unittest.html
.. _`Dependency injection`: http://www.youtube.com/watch?v=RlfLCWKxHJ0

**Version control:**

Commits should be reversible, independent units if possible. Use descriptive
titles and also add an explaining commit message unless the modification is
trivial. See also: `A Note About Git Commit Messages`_.

.. _`A Note About Git Commit Messages`: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

**Tests:**

Currently, two major test services are used:

- The tests on CDash_ are run on a daily basis on the ``master`` branch and
  on update of the ``testing`` branch. It ensures that the integration
  tests for the LHC models are working correctly on all tested platforms.
  The tests are run only on specific python versions.

- The `Travis CI`_ service is mainly used to check that the unit tests for
  pymad itself execute on several python versions. Python{2.6,2.7,3.3} are
  supported. The tests are executed on any update of an upstream branch.

.. _CDash: http://abp-cdash.web.cern.ch/abp-cdash/index.php?project=pymad
.. _`Travis CI`: https://travis-ci.org/pymad/cpymad


**Contribution work flow:**

All changes are reviewed via pull-requests. Before merging to master the
pull-request must reside aliased by the ``testing`` branch long enough to
be confirmed as stable.  Any issues are discussed in the associated issue
thread.  Concrete suggestions for changes are best posed as pull-requests
onto the feature branch.

