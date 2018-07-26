.. _troubleshooting:

Troubleshooting
---------------

In the following we will try to keep a list of the various issues and fixes
that might occur during or after installation.

After a successful installation, please use the following command to verify
that the extension can be loaded:

.. code-block:: bash

    python -c 'import cpymad.libmadx as l; l.start()'

The MAD-X banner should appear.


ImportError: undefined symbol: dgelsd\_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

    ImportError: /home/thomas/hit/dev/cpymad/cpymad/libmadx.cpython-36m-x86_64-linux-gnu.so: undefined symbol: dgelsd_

This problem can occur due to not specifying relevant link libraries, in
particular blas and/or lapack if they are available on your system (cpymad
does not try to detect if they are and just assumes they are not).

You can fix the problem by passing the appropriate libraries during the build
step::

    python setup.py build_ext -lblas -llapack


ImportError: libmadx.so
~~~~~~~~~~~~~~~~~~~~~~~

Message::

    ImportError: libmadx.so: cannot open shared object file: No such file or directory


This error occors if cpymad was linked against a dynamic library version of
MAD-X that can not be found at runtime. Reasons may be that the MAD-X
installation was moved or removed after building cpymad, or that ``RPATH`` was
not set properly on the cpymad extension. You can fix the problem by
specifying the correct ``RPATH`` to the setup script when building:

.. code-block:: bash

    python setup.py install --madxdir=<prefix>
    # or alternatively:
    python setup.py build_ext --rpath=<rpath>
    python setup.py install

Where ``<prefix>`` is the base folder containing the subfolders ``bin``,
``include``, ``lib`` of the MAD-X build and ``<rpath>`` contains the
dynamic library files.

If this does not work, you can set the ``LD_LIBRARY_PATH`` (or
``DYLD_LIBRARY_PATH`` on OSX) environment variable before running pymad, for
example:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/


OSError: Missing source file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

    OSError: Missing source file: 'cpymad/libmadx.c'. Install Cython to resolve this problem.

The easiest way to install Cython is:

.. code-block:: bash

    pip install cython

Alternatively, you can install pymad from the PyPI source distribution
which includes all source files.


Unable to find vcvarsall.bat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

    error: Unable to find vcvarsall.bat

Solution:
Open or create the file :file:`C:\\Python27\\Lib\\distutils\\distutils.cfg`
and add the following lines:

.. code-block:: cfg

    [build]
    compiler=mingw32

If you do not want to modify your python system configuration you can place
this as :file:`setup.cfg` in the current directory.

.. seealso:: http://stackoverflow.com/q/2817869/650222


unrecognized command line option '-mno-cygwin'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

    gcc: error: unrecognized command line option '-mno-cygwin'
    error: command 'gcc' failed with exit status 1

Solution:
In the file :file:`C:\\Python27\\Lib\\distutils\\cygwinccompiler.py` delete
every occurence of the string ``-mno-cygwin`` in the ``class
Mingw32CCompiler`` (about line 320). Depending on your version of
distutils, for example:

.. code-block:: diff

    @@ -319,11 +319,11 @@ class Mingw32CCompiler (CygwinCCompiler):
            else:
                entry_point = ''

    -       self.set_executables(compiler='gcc -mno-cygwin -O -Wall',
    -                            compiler_so='gcc -mno-cygwin -mdll -O -Wall',
    -                            compiler_cxx='g++ -mno-cygwin -O -Wall',
    -                            linker_exe='gcc -mno-cygwin',
    -                            linker_so='%s -mno-cygwin %s %s'
    +       self.set_executables(compiler='gcc -O -Wall',
    +                            compiler_so='gcc -mdll -O -Wall',
    +                            compiler_cxx='g++ -O -Wall',
    +                            linker_exe='gcc ',
    +                            linker_so='%s %s %s'
                                            % (self.linker_dll, shared_option,
                                                entry_point))
            # Maybe we should also append -mthreads, but then the finished

.. seealso:: http://stackoverflow.com/q/6034390/650222
