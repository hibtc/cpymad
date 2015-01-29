WinPython
=========

WinPython_ is a portable Python distribution which allows to build
self-contained C extensions. If you want to deploy your your CPyMAD build to
other machines, this is the distribution of choice.


Install dependencies
~~~~~~~~~~~~~~~~~~~~

Except for WinPython_ itself, you just need CMake_.


Build libmadx
~~~~~~~~~~~~~

Download and extract the `MAD-X source`_ from SVN. Open a terminal by
executing the :file:`%WinPython%\\WinPython Command Prompt.exe`. Change the directory to
the extracted MAD-X folder with the ``cd`` command and prepare the build:

.. code-block:: bat

    mkdir build
    cd build
    cmake -G "MinGW Makefiles" -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=..\install ..

In the file :file:`%MADX%\\build\\src\\CMakeFiles\\madxbin.dir\\link.txt`
search for ``-lgcc_eh`` and remove it, otherwise you may get linker errors at
the end of the next command. The following step will build the library. This
may take a few minutes, so go and grep a coffee meanwhile:

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

    python setup.py build --compiler=mingw32 --madxdir=<madx-install-path>
    python setup.py install

If the build step fails with linker error, try adding ``'quadmath'`` to the
``libaries`` variable in :file:`setup.py`.


.. _WinPython: http://winpython.sourceforge.net/
.. _CMake: http://www.cmake.org/
.. _MAD-X source: http://svnweb.cern.ch/world/wsvn/madx/tags/
.. _CPyMAD source: https://github.com/pymad/cpymad/zipball/master
