.. _troubleshooting:
Troubleshooting
---------------

In the following we will try to keep a list of the various issues and fixes
that might occur during installation.


ImportError: libmadx.so
~~~~~~~~~~~~~~~~~~~~~~~

Message::

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

Where ``<prefix>`` is the base folder containing the subfolders ``bin``,
``include``, ``lib`` of the MAD-X build and ``<rpath>`` contains the
dynamic library files.

If this does not work, you can set the LD_LIBRARY_PATH (or
DYLD_LIBRARY_PATH on OSX) environment variable before running pymad, for
example:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/


OSError: Missing source file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

    OSError: Missing source file: 'src/cern/cpymad/libmadx.c'. Install Cython to resolve this problem.

Solution:
The easiest way to install Cython is:

.. code-block:: bash

    pip install cython

Alternatively, you can install pymad from the PyPI source distribution
which includes all source files.


Unable to find vcvarsall.bat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occurs:
While building ``python setup.py install``.

Reason:
distutils is not configured to use MinGW.

Solution:
Add the following lines to :file:`C:\\Python27\\Lib\\distutils\\distutils.cfg`

.. code-block:: cfg

    [build]
    compiler=mingw32

If you do not want to modify your python system configuration you can place
this as :file:`setup.cfg` in the current directory. You can also specify
the compiler on the command line:

.. code-block:: bash

    python setup.py build --madxdir=<madx-install-path> --compiler=mingw32
    python setup.py install --madxdir=<madx-install-path>

See also `this question on stackoverflow <http://stackoverflow.com/questions/2817869/error-unable-to-find-vcvarsall-bat>`_.


TypeError: 'NoneType' object has no attribute '__getitem__'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Message::

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
While building ``python setup.py install``.

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
