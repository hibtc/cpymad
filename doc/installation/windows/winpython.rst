WinPython
=========

WinPython_ is a portable Python distribution which allows to build
self-contained C extensions. If you want to deploy your your CPyMAD build to
other machines, this is the distribution of choice.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

Except for WinPython_ itself, you just need CMake_.

Furthermore, cpymad now depends on minrpc_::

    pip install minrpc


Build libmadx
~~~~~~~~~~~~~

Download and extract the `MAD-X source`_ from SVN. Open a terminal by
executing the :file:`%WinPython%\\WinPython Command Prompt.exe`. Change the directory to
the extracted MAD-X folder with the ``cd`` command and prepare the build:

.. code-block:: bat

    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=..\install ..

In the file :file:`%MADX%\\build\\src\\CMakeFiles\\madxbin.dir\\link.txt` and
:file:`linklibs.rsp` search for ``-lgcc_eh`` and remove it (if present) -
otherwise you may get linker errors at the end of the next command.

The following step will build the library. This may take a few minutes, so go
and grep a coffee meanwhile:

.. code-block:: bat

    mingw32-make
    mingw32-make install

If all went well the last command will have installed binaries and library
files to the :file:`madX\\install` subfolder.


Build CPyMAD
~~~~~~~~~~~~

Download the `CPyMad source`_. Then go to the pymad folder and build as
follows:

.. code-block:: bat

    python setup.py build --madxdir=<madx-install-path>

You may encounter linker errors like the following::

    .../libgfortran.a(write.o):(.text$write_float+0xbb): undefined reference to `signbitq'
    .../lib/gcc/i686-w64-mingw32/4.9.2/libgfortran.a(write.o):(.text$write_float+0xe7): undefined reference to `finiteq'

If so, try the following instead:

.. code-block:: bat

    python setup.py build_ext --madxdir=<...> -lquadmath
    python setup.py build

From the built package you can create a so called wheel_, which is
essentially a zip archive containing all the files ready for installation:

.. code-block:: bat

    python setup.py bdist_wheel

This will create a ``.whl`` file named after the package and its target
platform. This file can now be used for installation in your favorite
python distribution, like so:

.. code-block:: bat

    pip install dist\cpymad-0.10.1-cp27-none-win32.whl


.. _WinPython: http://winpython.sourceforge.net/
.. _CMake: http://www.cmake.org/
.. _minrpc: https://pypi.python.org/pypi/minrpc
.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/tags/
.. _CPyMAD source: https://github.com/pymad/cpymad/zipball/master
.. _wheel: https://wheel.readthedocs.org/en/latest/
