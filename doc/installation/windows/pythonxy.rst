Python(x,y)
===========

I recommend `Python(x,y)`_ if you want a single ready-to-use python
installation with a large variety of packages and features. Make sure *Cython*
and *MinGW* are marked for installation.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

First install CMake_. In the *Install options* dialog choose to *Add CMake
to the system PATH for all/current user* according to your liking.

.. note::

    You can also select *Do not add CMake to the system PATH*. In this case
    you will need to manually extend your PATH later on in the terminal:

    .. code-block:: bat

        set PATH=%PATH%;C:\Program Files (x86)\CMake\bin



Build libmadx
~~~~~~~~~~~~~

Download and extract the latest `MAD-X source`_ tag from SVN.

.. note::

    If the latest release contains bugs that have been fixed in the
    meantime you may want to download the `latest revision`_ from SVN. Be
    aware though, that the latest revision is not at all guaranteed to be
    stable or even buildable.

.. note::

    You might need multiple extraction steps until you get a folder
    containing the file :file:`CMakeLists.txt` as well as several other
    files and subdirectories.

I recommend to build MAD-X as a *static* library as described below. This
way, you won't need to carry any ``.dll`` files around and you won't run
into version problems when having a multiple MAD-X library builds around.

`Python(x,y)`_ provides an *Open enhanced console here* context menu item
when right-clicking on a folder in the explorer. Open the madX folder with
it and run the commands:

.. code-block:: bat

    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=..\install ..
    make
    make install

This will install the headers, binaries and library files to the
:file:`madX\\install` subfolder.

.. note::

    You can also use the regular console ``cmd.exe``. In this case extend
    the PATH environment variable with your python and MinGW installation:

    .. code-block:: bat

        set PATH=%PATH%;C:\MinGW32-xy\bin
        set PATH=%PATH%;C:\Python27
        set PATH=%PATH%;C:\Python27\Scripts

.. note::

    If executing CMake from the GUI, you have to disable the
    ``BUILD_SHARED_LIBS`` option, if present. Afterwards reconfigure and
    regenerate.


Build CPyMAD
~~~~~~~~~~~~

Download the `CPyMad source`_. Then go to the pymad folder and build as
follows:

.. code-block:: bat

    python setup.py build --compiler=mingw32 --madxdir=<madx-install-path>
    python setup.py install

Substitute ``<madx-install-path>`` with the :file:`madX\\install` subfolder
as specified by ``CMAKE_INSTALL_PREFIX`` before.


.. _Python(x,y): https://code.google.com/p/pythonxy/
.. _CMake: http://www.cmake.org/
.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/tags/
.. _latest revision: http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1
.. _CPyMAD source: https://github.com/pymad/cpymad/zipball/master
