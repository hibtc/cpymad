Windows
-------

Pre-built binaries
~~~~~~~~~~~~~~~~~~

On a windows platform, you are likely to run into more problems than on
linux when building from source. Therefore, we will try to provide `built
versions`_ for some platforms. If your platform is supported, you can just
run

.. code-block:: bat

    pip install cern-cpymad

from your terminal and that's it. If this fails, you will need to build
from source:


Install dependencies
~~~~~~~~~~~~~~~~~~~~

For source builds I recommend `Python(x,y)`_. This is a python distribution
which includes all :ref:`dependencies` except for CMake_. Make sure Cython
and MinGW are marked for installation.

.. _CMake: http://www.cmake.org/


Install libmadx
~~~~~~~~~~~~~~~

Download and extract the `MAD-X source`_ from SVN.

I recommend building MAD-X as a *static* library. This way, you won't
need to carry any ``.dll`` files around and you won't run into version
problems when having a multiple MAD-X library builds around.

`Python(x,y)`_ provides an *Open enhanced console here* context menu item
when right-clicking on a folder in the explorer. Open the madX folder with
it (or with ``cmd.exe``) and run the commands:

.. code-block:: bat

    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=..\install ..
    make install

This will install the headers, binaries and library files to the
:file:`madX/install` subfolder.

If executing CMake from the GUI, you have to disable the
``BUILD_SHARED_LIBS`` option, if present. Afterwards reconfigure and
regenerate.


Install CPyMAD
~~~~~~~~~~~~~~

Download the `CPyMad source`_. Then go to the pymad folder and build as
follows:

.. code-block:: bat

    python setup.py install --madxdir=<madx-install-path>


.. _built versions: https://pypi.python.org/pypi/cern-cpymad/0.7
.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1
.. _CPyMAD source: https://github.com/pymad/cpymad/zipball/master
.. _Python(x,y): https://code.google.com/p/pythonxy/
