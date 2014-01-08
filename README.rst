PyMAD
-----

PyMAD is a python wrapper for MAD-X_ (Methodical Accelerator Design). It can
be used in either of two modes:

- cpymad uses Cython_ to access the Mad-X_ library directly
- jpymad uses JMad_ via Py4J_ which then accesses the Mad-X_ library

.. _MAD-X: http://madx.web.cern.ch/madx/
.. _Cython: http://cython.org/
.. _JMad: http://jmad.web.cern.ch/jmad/
.. _Py4J: http://py4j.sourceforge.net/

Prerequisites
~~~~~~~~~~~~~

General
=======

- python with numpy

CPyMAD
======

- Cython_
- gfortran or similar
- MAD-X_ shared library
- ``madextern.h`` in your include path

JPyMAD
======

- Java
- JMad_
- Py4J_

Installation
~~~~~~~~~~~~

Further instructions are available at http://cern.ch/pymad/installation.


Usage
~~~~~

.. code-block:: python

    # Once installed this is a nice example showing current
    # usability (run from examples folder):
    from cern import pymad

    # select backend,
    # 'cpymad' is currently default if nothing is provided
    # Returns a pymad.service object
    pms = pymad.init('jpymad')

    # Create a model:
    pm = pms.create_model('lhc')

    # Run twiss:
    # This returns a "lookup dictionary" containing
    # numpy arrays. Lowercase keys.
    twiss,summary = pm.twiss()

    # Your own analysis below:
    import matplotlib.pyplot as plt
    plt.plot(twiss.s, twiss.betx)


See http://cern.ch/pymad/ for further documentation.

