Known issues
~~~~~~~~~~~~

- The MAD-X command output may sometimes appear delayed and in wrong order.
  This is a problem with mixing output from C and Fortran code. On linux it
  can be fixed by setting ``export GFORTRAN_UNBUFFERED_PRECONNECTED=y`` in the
  environment. On windows, this can be tried as well, but is not reliable to
  our knowledge.

- **windows + python 3.3:** there is currently no satisfying way to close file
  handles in the MAD-X process or prevent them from being inherited by
  default.  You have to make sure on your own that you close all file handles
  before creating a new ``cpymad.madx.Madx`` instance!

- the MAD-X ``USE`` command invalidates table row names. Therefore, using
  ``Table.dframe()`` is unsafe after ``USE`` should be avoided, unless
  manually specifying an index, e.g. ``Table.dframe(index='name')``, see `#93`_.

.. _#93: https://github.com/hibtc/cpymad/issues/93
