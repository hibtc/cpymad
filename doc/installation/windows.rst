Windows
-------

Prebuilt binaries
=================

On windows, I provide `built versions`_ for some python versions. If your
platform is supported, you can just run

.. code-block:: bat

    pip install cpymad

from your terminal and that's it. In case of problems please read the rest of
this section.


Offline installation
====================

If you want to install to a target machine without internet access (or with
firewall), you can manually download_ the ``.whl`` file for your platform and
then use pip_ to install that particular file. For example:

.. code-block:: bat

    pip install cpymad-0.17.3-cp27-none-win32.whl

In this case you will also need to grab and install *numpy* and
*setuptools* (and possibly their dependencies if any). Installable wheel_
archives for these packages can be conveniently downloaded from here:
`Unofficial Windows Binaries for Python Extension Packages`_.

To simplify this process, ``pip`` can automatically download all required
files:

.. code-block:: bat

    mkdir packages
    pip download -d packages cpymad

and then later install them in the offline environment like this:

.. code-block:: bat

    pip install -f packages cpymad

If you want/have to build from source, have a look at following guide.

.. _built versions: https://pypi.python.org/pypi/cpymad/#downloads
.. _download: https://pypi.python.org/pypi/cpymad/#downloads
.. _pip: https://pypi.python.org/pypi/pip
.. _wheel: https://wheel.readthedocs.org/en/latest/
.. _Unofficial Windows Binaries for Python Extension Packages: http://www.lfd.uci.edu/~gohlke/pythonlibs/


Manual build
============

In order to build cpymad manually on python≤3.4, I recommend WinPython_. This
portable python distribution includes a MinGW compiler suite that allows to
create self-contained binaries which can later be used on other python
distributions as well. Python≥3.5 is currently unsupported due to a change in
python's windows build process.

.. _WinPython: http://winpython.github.io/


Install dependencies
~~~~~~~~~~~~~~~~~~~~

Except for WinPython_ itself, you just need CMake_ and minrpc_.

First install CMake_. In the *Install options* dialog choose to *Add CMake
to the system PATH for all/current user* according to your liking.
Alternatively, manually extend your PATH later in the terminal:

.. code-block:: bat

    set PATH=%PATH%;C:\Program Files (x86)\CMake\bin

Next, install minrpc_ and cython_::

    pip install minrpc cython

.. _CMake: http://www.cmake.org/
.. _minrpc: https://pypi.python.org/pypi/minrpc
.. _cython: https://cython.org


Build libmadx
~~~~~~~~~~~~~

Download and extract the `latest MAD-X release`_.

.. _latest MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases

I recommend to build MAD-X as a *static* library as described below. This
way, you won't need to carry any ``.dll`` files around and you won't run
into version problems when having a multiple MAD-X library builds around.

Open a terminal by executing the :file:`%WinPython%\\WinPython Command
Prompt.exe`. Change the directory to the extracted MAD-X folder with the
``cd`` command and prepare the build:

.. code-block:: bat

    mkdir build
    cd build
    cmake .. ^
        -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DMADX_STATIC=ON ^
        -DMADX_INSTALL_DOC=OFF ^
        -DCMAKE_INSTALL_PREFIX=..\install

In the file :file:`%MADX%\\build\\src\\CMakeFiles\\madxbin.dir\\link.txt` and
:file:`linklibs.rsp` search for ``-lgcc_eh`` and remove it (if present) -
otherwise you may get linker errors at the end of the next command.

The following step will build the library. This may take a few minutes, so go
and grab a coffee meanwhile:

.. code-block:: bat

    mingw32-make
    mingw32-make install

If all went well the last command will have installed binaries and library
files to the :file:`%MADX%\\install` subfolder.


Build cpymad
~~~~~~~~~~~~

Download and extract the latest `cpymad release`_. Alternatively, use git to
retrieve the current development version (unstable):

.. code-block:: bat

    git clone https://github.com/hibtc/cpymad

Then go to the cpymad folder and build as follows:

.. code-block:: bat

    python setup.py build_ext --static --madxdir=<madx-install-path>
    python setup.py build

From the built package you can create a so called wheel_, which is
essentially a zip archive containing all the files ready for installation:

.. code-block:: bat

    python setup.py bdist_wheel

This will create a ``.whl`` file named after the package and its target
platform. This file can now be used for installation in your favorite
python distribution, like so:

.. code-block:: bat

    pip install dist\cpymad-0.17.3-cp27-none-win32.whl

.. _cpymad release: https://github.com/hibtc/cpymad/releases
.. _wheel: https://wheel.readthedocs.org/en/latest/
