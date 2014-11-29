Unix
----

On Unix-like system building CPyMAD from source is most convenient and
should be unproblematic.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

On many distributions all :ref:`dependencies` can be installed from the
official repositories.


Install libmadx
~~~~~~~~~~~~~~~

Download the `MAD-X source`_ from SVN

.. code-block:: bash

    svn co http://svn.cern.ch/guest/madx/trunk/madX

Build and install the library

.. code-block:: bash

    mkdir build && cd build
    cmake -DMADX_STATIC=OFF -DBUILD_SHARED_LIBS=ON ../madX
    make && make install

The install step might require root privileges if not changing the
installation prefix.


Install CPyMAD
~~~~~~~~~~~~~~

The latest `CPyMAD release`_ can conveniently be installed using pip_:

.. code-block:: bash

    pip install cpymad

If you run into problems with this, you should manually download and
install the `CPyMad source`_ to see in which step the problem occurs:

.. code-block:: bash

    git clone git://github.com/hibtc/cpymad
    cd pymad
    python setup.py build
    python setup.py install

You might need root privileges for the last step if not installing to a
virtualenv_.


.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1
.. _CPyMAD release: https://pypi.python.org/pypi/cpymad
.. _pip: https://pypi.python.org/pypi/pip
.. _CPyMAD source: https://github.com/hibtc/cpymad/zipball/master
.. _virtualenv: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
