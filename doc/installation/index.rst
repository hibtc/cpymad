.. role:: bash(code)
   :language: bash

Installation Instructions
*************************

Installation of PyMad is different depending on which underlying modules you want.


JPyMad
------

For JPyMad, you first need JMad which you can obtain `here <http://cern.ch/jmad/>`_
Then, download the `pymad source <https://github.com/pymad/pymad>`_, and in pymad/src, run
the command

.. code-block:: bash

    python setup.py install --install-platlib

The argument *--install-platlib* means we exclude external modules. The reason is that the external module cern.madx requires the dynamic library of Mad-X available on your system (which we here assume you do not have)

CPyMad (unix)
-------------

First method, use installation script:
    We provide an `installation script <install.sh>`_ which should do the full job for you. Download the script
    and run it. It will take a few minutes to finish. Upon successful completion, it will create an uninstall.sh
    script which you should keep somewhere. This script knows all the files you have installed,
    and will remove all files if you execute it (folders are not removed).

    Dependencies: cmake, compilers for c/fortran, python 2.6 or 2.7

Second method, source script:

  For CPymad on a 64 bit Linux machine with afs available, it is enough to source the script that we
  currently provide in

   /afs/cern.ch/user/y/ylevinse/public/setupCPYMAD.sh

  We hope to in the near future provide a tarball for Linux and OSX containing all needed files.

Third method, manual installation:

    * Download the Mad-X source from
      `svn <http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1>`_
      and unpack it.
    * Enter the folder madX and run the commands

      .. code-block:: bash

          mkdir build; cd build
          cmake -DMADX_STATIC=OFF -DBUILD_SHARED_LIBS=ON ../
          make install

    * Download the PyMad source from `github <https://github.com/pymad/pymad/zipball/master>`_
      and unpack it
    * In the folder pymad/src, run the command

      .. code-block:: bash

          python setup.py install

If you download JMad after following any of the methods described above for CPyMad,
you will immediately have JPyMad available as well.


Cython (windows, MinGW)
-----------------------

At this time you have to build pymad manually.

    * If you want to install all system dependencies at once, I recommend `Python(x,y) <https://code.google.com/p/pythonxy/>`_. This is a python development distribution including MinGW, Cython and Python2.7. Make sure Cython and MinGW are marked for installation.

    * Download the Mad-X source from
      `svn <http://svnweb.cern.ch/world/wsvn/madx/trunk/madX/?op=dl&rev=0&isdir=1>`_
      and unpack it.

    * I recommend building MAD-X as a *static* library. This way, you won't
      need to carry any ``.dll`` files around and you won't run into version
      problems when having a multiple MAD-X library builds around.

      Enter the folder madX and run the commands

      .. code-block:: bat

          mkdir build && cd build
          cmake -G "MinGW Makefiles" -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_INSTALL_PREFIX=..\madx-redist ..\
          make install

      This will install the headers, binaries and library files to the folder ``..\madx-redist``.

      Executing CMake from the GUI, you have to disable the ``BUILD_SHARED_LIBS`` option, if present. Afterwards reconfigure and regenerate.

    * In the folder ``pymad/src``, run the command

      .. code-block:: bat

          python setup.py install --madxdir=<path-to-your>\madx-redist

      It is highly unlikely that your build succeeds at this point. See :ref:`potential-problems` for further information.



.. _potential-problems:

Potential problems
------------------

In the following we will try to keep a list of the various issues users have reported during installation.

    * libmadx.so not found::

          from cern.madx import madx
          ImportError: libmadx.so: cannot open shared object file: No such file or directory

      Reason:
      The runtime path for the MAD-X static library is configured
      incorrectly.

      Solution:
      You can pass the correct path to the setup script when building:

      .. code-block:: bash

         python setup.py install --madxdir=<prefix>
         # or alternatively:
         python setup.py build_ext --rpath=<rpath>
         python setup.py install

      Where ``<prefix>`` is the base folder, containing the subfolders
      ``bin``, ``include``, ``lib`` of the MAD-X build and ``<rpath>``
      contains the dynamic library files.

      If this does not work, you can set the LD_LIBRARY_PATH (or
      DYLD_LIBRARY_PATH on OSX) environment variable before running pymad,
      for example:

      .. code-block:: bash

          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/

    * Can't copy 'src/cern/\*\*/\*.c':

      ::

        error: can't copy 'src/cern/libmadx/table.c': doesn't exist or not a regular file

     Solution:
     If installing from the repository, you need Cython. The easiest way to
     install Cython is:

     .. code-block:: bash

        pip install cython

     In order to get cpymad, you need Cython installed on your system. If you cannot obtain that, use jpymad instead.

    * Unable to find vcvarsall.bat:

      Occurs:
      While building :bash:`python setup.py install`.

      Reason:
      distutils is not configured to use MinGW.

      Solution:
      Add the following lines to :file:`C:\\Python27\\Lib\\distutils\\distutils.cfg``

      .. code-block:: cfg

        [build]
        compiler=mingw32


      If you do not want to modify your python system configuration you can place this as :file:`setup.cfg` in the current directory. You can also specify the compiler on the command line:

      .. code-block:: bat

        python setup.py build --madxdir=<path-to-your>\madx-redist --compiler=mingw32
        python setup.py install --madxdir=<path-to-your>\madx-redist


      See also `this question on stackoverflow <http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat>`_.

    * distutils.unixcompiler not configured:

      .. code-block:: python

        Traceback (most recent call last):
          ...
          File "C:\Python27\lib\distutils\unixccompiler.py", line 227, in runtime_library_dir_option
            compiler = os.path.basename(sysconfig.get_config_var("CC"))
          File "C:\Python27\lib\ntpath.py", line 198, in basename
            return split(p)[1]
          File "C:\Python27\lib\ntpath.py", line 170, in split
            d, p = splitdrive(p)
          File "C:\Python27\lib\ntpath.py", line 125, in splitdrive
            if p[1:2] == ':':
        TypeError: 'NoneType' object has no attribute '__getitem__'

      Occurs:
      While building :bash:`python setup.py install`.

      Reason:
      Bug in distutils (?).

      Solution:
      Add the following line to :file:`C:\\Python27\\Lib\\distutils\\sysconfig.py`:

      .. code-block:: python
        :emphasize-lines: 5

        def _init_nt():
            """Initialize the module as appropriate for NT"""
            g = {}
            ...
            g['CC'] = 'gcc'
            ...
            _config_vars = g

      For further reference see `a related issue <http://bugs.python.org/issue2437>`_.
