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
from source. The following guide assumes that you use *MinGW* as compiler
suite.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

I recommend to use `Python(x,y)`_. This is a python distribution which
includes all :ref:`dependencies` except for *CMake*. Make sure *Cython* and
*MinGW* are marked for installation.

.. note::

    If you want to use another python installation things will get more
    complicated. I recommend the following resources:

    * MinGW_, since you need a compiler to build *libmadx* and *CPyMAD*
    * `Unofficial Windows Binaries for Python Extension Packages`_
    * `Compiling Python extensions with distutils and MinGW`_


Now install CMake_. In the *Install options* dialog choose to *Add CMake
to the system PATH for all/current user* according to your liking.

.. note::

    You can also select *Do not add CMake to the system PATH*. In this case
    you will need to manually extend your PATH later on in the terminal:

    .. code-block:: bat

        set PATH=%PATH%;C:\Program Files (x86)\CMake\bin


Build libmadx
~~~~~~~~~~~~~

Download and extract the `MAD-X source`_ from SVN.

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
:file:`madX/install` subfolder.

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

    python setup.py install --madxdir=<madx-install-path>

Substitute ``<madx-install-path>`` with the :file:`madX\\install` subfolder
as specified by ``CMAKE_INSTALL_PREFIX`` before.


.. _built versions: https://pypi.python.org/pypi/cern-cpymad/
.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/tags/
.. _latest revision: http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1
.. _CPyMAD source: https://github.com/pymad/cpymad/zipball/master
.. _Python(x,y): https://code.google.com/p/pythonxy/
.. _CMake: http://www.cmake.org/
.. _MinGW: http://www.mingw.org/
.. _Unofficial Windows Binaries for Python Extension Packages: http://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _Compiling Python extensions with distutils and MinGW: http://eli.thegreenplace.net/2008/06/28/compiling-python-extensions-with-distutils-and-mingw/
