Unix
----

On Unix-like system building cpymad from source is most convenient and
should be unproblematic.


.. _dependencies:

Install dependencies
~~~~~~~~~~~~~~~~~~~~

On many linux distributions most or all of the following dependencies will
be available in the official repositories. Use these as the preferred
install method.

To build MAD-X and cpymad from source you will need

- CMake_
- Python>=2.7
- C / Fortran compilers (gcc >= 4.8)

Furthermore, cpymad depends on the following python packages:

- setuptools_
- Cython_
- NumPy_
- PyYAML_
- minrpc_

The python packages can be installed using pip_.

.. _CMake: http://www.cmake.org/
.. _setuptools: https://pypi.python.org/pypi/setuptools
.. _Cython: http://cython.org/
.. _NumPy: http://www.numpy.org/
.. _PyYAML: https://pypi.python.org/pypi/PyYAML
.. _pip: https://pypi.python.org/pypi/pip
.. _minrpc: https://pypi.python.org/pypi/minrpc


Install libmadx
~~~~~~~~~~~~~~~

Download the `latest MAD-X release`_ from github or use directly the source
code from master (unstable):

.. code-block:: bash

    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X

Build and install the library

.. code-block:: bash

    mkdir build && cd build
    cmake ../madX \
        -DMADX_STATIC=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DUSE_GC=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN'
    make && make install

The install step might require root privileges if not changing the
installation prefix.


Install cpymad
~~~~~~~~~~~~~~

The latest `cpymad release`_ can conveniently be installed using pip_:

.. code-block:: bash

    pip install cpymad

If you run into problems with this, you should manually download and
install the cpymad_ source_ to see in which step the problem occurs:

.. code-block:: bash

    git clone https://github.com/hibtc/cpymad
    cd cpymad
    python setup.py build
    python setup.py install

You might need root privileges for the last step if not installing to a
virtualenv_.


.. _latest MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases
.. _cpymad release: https://pypi.python.org/pypi/cpymad#downloads
.. _pip: https://pypi.python.org/pypi/pip
.. _cpymad: https://github.com/hibtc/cpymad
.. _source: https://github.com/hibtc/cpymad/zipball/master
.. _virtualenv: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
