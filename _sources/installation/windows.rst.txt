.. highlight:: batch

Windows
-------

cpymad is linked against a library version of MAD-X, which means that in order
to build cpymad you first have to compile MAD-X from source. The official
``madx`` executable is not sufficient. These steps are described in the
following subsections:

.. contents:: :local:


Setup environment
=================

Setting up a functional build environment requires more effort on windows than
on other platforms, and there are many pitfalls if not using the right
compiler toolchain (such as linking against DLLs that are not present on most
target systems).

We recommend that you install **conda**. It has proven to be a reliable tool
for this job. Specifically, I recommend getting Miniconda_; anaconda should
work too, but I wouldn't recommend it because it ships many unnecessary
components.

Note that while conda is used to setup a consistent environment for the build
process, the generated cpymad build will be usable with other python
distributions as well.

.. _miniconda: https://conda.io/en/latest/miniconda.html


**Install build tools:**

After installing conda, open the conda command prompt.

We show the commands for the case of working inside a ``cmd.exe`` terminal
(batch). If you prefer to work with powershell or bash on msys2, the workflow
(commands/arguments) will be mostly the same, with the only exception that you
have to adapt the syntax accordingly in some places.

Now create a build environment with cmake_ and setup the MinGW compiler
toolchain::

    conda create -n buildenv
    conda activate buildenv

    conda install -c anaconda cmake
    conda install -c msys2 m2w64-toolchain

.. _cmake: http://www.cmake.org/


**Get the sources:**

Now download and extract the latest `MAD-X release`_ and  `cpymad release`_
side by side.

Alternatively, use git if you want to build a development version from local
checkout (unstable)::

    conda install git
    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X
    git clone https://github.com/hibtc/cpymad

.. _MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases
.. _cpymad release: https://github.com/hibtc/cpymad/releases


Build MAD-X
===========

There are two major alternatives how to build MAD-X:

- I recommend to build MAD-X as a *static* library as described below. This
  way, you won't need to carry any ``.dll`` files around and you won't run
  into version problems when having a multiple MAD-X library builds around.

- However, in some cases it may be easier to link cpymad *dynamically* against
  MAD-X, i.e. using a DLL. This comes at the cost of having to redistribute
  the ``madx.dll`` along. The choice is yours.

Note that if you're planning on using MSVC to build the cpymad extension later
on, you have to build MAD-X as shared library (DLL). With MinGW (recommended),
both build types are supported.

In the following, we show the commands for the static build.

In the build environment, type::

    mkdir MAD-X\build
    cd MAD-X\build

    cmake .. -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DMADX_STATIC=ON ^
        -DCMAKE_INSTALL_PREFIX="../dist"

    cmake --build . --target install

.. note::

    For shared library builds, instead use ``-DBUILD_SHARED_LIBS=ON``.

If all went well the last command will have installed binaries and library
files to the ``MAD-X\dist`` subfolder.

Save the path to this install directory in the ``MADXDIR`` environment
variable. This variable will be used later by the ``setup.py`` script to
locate the MAD-X headers and library, for example::

    set "MADXDIR=C:\Users\<....>\MAD-X\dist"

Also, set the following variables according to the flags passed to the cmake
command above::

    set STATIC=1    # if -DMADX_STATIC=ON
    set SHARED=1    # if -DBUILD_SHARED_LIBS=ON


Build cpymad
============

Using MinGW
~~~~~~~~~~~

For building cpymad you can simply reuse the build environment with the MinGW
installation from above, and now also install the targeted python version into
the build environment, e.g.::

    conda install python=3.7 wheel cython

.. note::

    If you want to build wheels for multiple python versions, just create a
    build environment for each target version, and don't forget to install the
    same version of the m2w64 compiler toolchain.

Now invoke the following command, which will cythonize the ``.pyx`` cython
module to ``.c`` code as a side effect::

    python setup.py build_py

Now comes the tricky part: we will have to manually build the C extension
using gcc, because setuptools doesn't know how to properly use our MinGW.

First set a few environment variables corresponding to the target platform
and python version::

    set py_ver=37
    set file_tag=cp37-win_amd64
    set dir_tag=win-amd64-cpython-37

On python 3.6 or earlier use the form ``set dir_tag=win-amd64-3.6`` instead.

With these values set, you should be able to copy-paste the following
commands::

    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set libdir=build\lib.%dir_tag%\cpymad

    mkdir %tempdir%
    mkdir %libdir%

    call %gcc% -mdll -O -Wall -DMS_WIN64 ^
        -I %MADXDIR%\include ^
        -I %pythondir%\include ^
        -c src/cpymad/libmadx.c ^
        -o %tempdir%\libmadx.obj ^
        -std=gnu99

    call %gcc% -shared -s ^
        %tempdir%\libmadx.obj ^
        -L %MADXDIR%\lib ^
        -lmadx -lDISTlib -lptc -lgc-lib -lstdc++ -lgfortran ^
        -lquadmath %pythondir%\python%py_ver%.dll -lmsvcr100 ^
        -o %libdir%\libmadx.%file_tag%.pyd

For old versions of MAD-X, leave out ``-lDISTlib`` from the second gcc call.

If this succeeds, you have most of the work behind you.

At this point, you may want to check the built ``.pyd`` file with `Dependency
Walker`_ to verify that it depends only on system dependencies (except for
``pythonXY.dll``, and in the case of dynamic linking ``madx.dll``).

We now proceed to build a so called wheel_. Wheels are zip archives containing
all the files ready for installation, as well as some metadata such as version
numbers etc. The wheel can be built as follows::

    python setup.py bdist_wheel

The ``.whl`` file is named after the package and its target platform. This
file can now be used for installation on this or any other machine running the
same operating system and python version. Install as follows::

    pip install dist\cpymad-*.whl

If you plan on changing cpymad code, do the following instead::

    pip install -e .

Finally, do a quick check that your cpymad installation is working by typing
the following::

    python -c "import cpymad.libmadx as l; l.start()"

The MAD-X startup banner should appear. You can also run more tests as
follows::

    python test\test_madx.py
    python test\test_util.py

Congratulations, you are now free to delete the MAD-X and cpymad folders (but
keep your wheel!).

.. _Dependency Walker: https://www.dependencywalker.com/
.. _wheel: https://wheel.readthedocs.org/en/latest/


Using Visual Studio
~~~~~~~~~~~~~~~~~~~

Python's official binaries are all compiled with the Visual C compiler and
therefore this is the only *officially* supported method to build python C
extensions on windows.

It is possible to build the cpymad C extension with Visual Studio, but there
is a good reason that the above guide doesn't use it:

Visual Studio doesn't include a Fortran compiler which means that you still
have to build MAD-X as described. Also, you have to build MAD-X as a shared
library, because the static library created by MinGW most likely won't be
compatible with the Visual C compiler.

First, look up `the correct Visual Studio version`_ and download and install
it directly from microsoft. It is possible that older versions are not
supported anymore.

.. _the correct Visual Studio version: https://wiki.python.org/moin/WindowsCompilers#Which_Microsoft_Visual_C.2B-.2B-_compiler_to_use_with_a_specific_Python_version_.3F

After that, activate the Visual Studio tools by calling ``vcvarsall.bat``.
Depending on your Visual Studio version and install path, this might look like
this::

    call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"

Once you've accomplished that, the steps to build cpymad should actually be
relatively simple (simpler than using MinGW in conda)::

    conda create -n py37 python=3.7
    conda activate py37
    conda install wheel cython
    python setup.py build_ext --shared --madxdir=%MADXDIR%
