Unix
----

On Unix-like system building CPyMAD from source is most convenient and
should be unproblematic.

On many distributions all dependencies can be installed from the official
repositories.


Install libmadx
~~~~~~~~~~~~~~~

Download the `MAD-X source`_ from SVN

.. code-block:: bash

    svn co http://svnweb.cern.ch/guest/madx/trunk/madX

Build and install the library

.. code-block:: bash

    mkdir build && cd build
    cmake -DMADX_STATIC=OFF -DBUILD_SHARED_LIBS=ON ../madX
    make && make install

The last step might require root privileges if not changing the
installation prefix.


Install CPyMAD
~~~~~~~~~~~~~~

The latest CPyMAD release can conveniently be installed using pip:

.. code-block:: bash

    pip install cern-pymad

If you run into problems with this, you should manually download and
install the `CPyMad source`_ to see in which step the problem occurs:

.. code-block:: bash

    git clone git://github.com/pymad/pymad
    cd pymad
    python setup.py build
    python setup.py install

You might need root privileges for the last step if not installing to a
virtualenv_.


.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1
.. _CPyMAD source: https://github.com/pymad/pymad/zipball/master
.. _virtualenv: http://virtualenv.readthedocs.org/en/latest/virtualenv.html
