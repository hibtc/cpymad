.. highlight:: bat

Windows
-------

On windows, I provide `built versions`_ for some python versions. If your
platform is supported, you can just run::

    pip install cpymad

In case of problems, or if you plan on changing cpymad code, please read the
rest of this page. You will have to manually compile first MAD-X and then
cpymad.

.. _built versions: https://pypi.python.org/pypi/cpymad/#downloads


Preparations
============

In order to build cpymad manually, I recommend using **conda** which makes it
very easy to install different python versions, quickly setup and tear down
independent environments and even acquire and build non-python dependencies
and compilers! Miniconda_ is perfectly suitable for all our intents, but
anaconda should work fine too.

.. _miniconda: https://conda.io/en/latest/miniconda.html

After installing conda, open the conda command prompt. In order to build
MAD-X, first **create a python 3.4** environment with cmake_ and mingwpy_ --
even if you plan using cpymad on a higher python version! If you want cpymad
on py2.7 or 3.3 you can use the target python version for building MAD-X
instead of 3.4::

    conda create -n py34 python=3.4 cmake
    conda activate py34
    conda install -c conda-forge mingwpy

.. _cmake: http://www.cmake.org/
.. _mingwpy: https://mingwpy.github.io/

Now download and extract the latest `MAD-X release`_ and  `cpymad release`_,
Alternatively, use git to retrieve the current development version
(unstable)::

    conda install git
    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X
    git clone https://github.com/hibtc/cpymad

.. _MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases
.. _cpymad release: https://github.com/hibtc/cpymad/releases

In the following there are two alternatives how to build MAD-X:

- I recommend to build MAD-X as a *static* library as described below. This
  way, you won't need to carry any ``.dll`` files around and you won't run
  into version problems when having a multiple MAD-X library builds around.

- However, it is somewhat easier to link cpymad *dynamically* against MAD-X on
  python versions above 3.4, i.e. using a DLL. This comes at the cost of
  having to redistribute the ``madx.dll`` along. The choice is yours.


Static build (recommended)
==========================

In the **python 3.4** environment, type::

    mkdir MAD-X\build-static
    cd MAD-X\build-static

    cmake .. ^
        -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DMADX_STATIC=ON ^
        -DMADX_INSTALL_DOC=OFF ^
        -DCMAKE_INSTALL_PREFIX=..\install

    mingw32-make install

The final step will build the library. This may take a few minutes, so go
and grab a coffee meanwhile.

If all went well the last command will have installed binaries and library
files to the :file:`%MADX%\\install` subfolder.

Save the **absolute path** to this install directory in the ``MADXDIR``
enviroment variable, this variable will be used later by the ``setup.py``
script to locate the MAD-X headers and library, for example::

    set "MADXDIR=C:\Users\<....>\MAD-X\install"


Targetting py35 or above
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use cpymad on a python version later than 3.4, read here,
otherwise skip to the next section.

Create an environment with your target python version, e.g.::

    conda create -n py37 python=3.7 wheel cython
    conda activate py37

Now comes the tricky part, you will will have to "cross-compile" (sort of) the
cython extension on the target platform with GCC from mingwpy in python 3.4.

First, set a few environment variables with the path of GCC, the python prefix
of the target python version and certain platform/abi tags. For a 64bit
python 3.7 this would look as follows::

    set py_ver=37
    set dir_tag=win-amd64-3.7
    set file_tag=cp37-win_amd64

And use this for good as follows::

    for /f %G in ('python -c "import sys; print(sys.prefix)"') do (
        set "gcc=%~fG\..\py34\Scripts\gcc.exe"
        set "pythondir=%~fG"
    )

    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set libdir=build\lib.%dir_tag%\cpymad

    mkdir %tempdir%
    mkdir %libdir%

    :: This will cythonize `.pyx` to `.c`:
    call python setup.py build_py

    call %gcc% -mdll -O -Wall ^
        -I%MADXDIR%\include ^
        -I%pythondir%\include ^
        -c src/cpymad/libmadx.c ^
        -o %tempdir%\libmadx.obj ^
        -std=gnu99

    :: Linking directly against the `pythonXX.dll` is the only way I found to
    :: satisfy the linker in a conda python environment. The conventional
    :: command line `-L%pythondir%\libs -lpython%py_ver%` used to work fine on
    :: WinPython, but fails on conda with large number of complaints about
    :: about undefined references, such as `__imp__Py_NoneStruct`,
    call %gcc% -shared -s ^
        %tempdir%\libmadx.obj ^
        -L%MADXDIR%\lib ^
        -lmadx -lptc -lgc-lib -lstdc++ -lgfortran ^
        -lquadmath %pythondir%\python%py_ver%.dll -lmsvcr100 ^
        -o %libdir%\libmadx.%file_tag%.pyd

Now skip to the final topic: Installation_.


Targetting py34 or below
~~~~~~~~~~~~~~~~~~~~~~~~

This works only if you are planning to **use** cpymad on an old python
version, on 3.4 or below.

Make sure that you are in a conda environment with the targeted python version
and type::

    conda install wheel cython
    python setup.py build_ext -c mingw32 --static --madxdir=%MADXDIR%

If this worked, go to the final topic: Installation_.


Dynamic build (easier)
======================

The DLL build works very similar, with a few minor differences. Type the
following::

    mkdir MAD-X\build-shared
    cd MAD-X\build-shared

    cmake .. ^
        -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=ON ^
        -DMADX_STATIC=OFF ^
        -DMADX_INSTALL_DOC=OFF ^
        -DCMAKE_INSTALL_PREFIX=..\install

    mingw32-make install

If all went well the last command will have installed binaries and library
files to the :file:`%MADX%\\install` subfolder.

Save the **absolute path** to the install directory in the ``MADXDIR``
enviroment variable, this variable will be used later by the ``setup.py``
script to locate the MAD-X headers and library. For example::

    set "MADXDIR=C:\Users\<....>\MAD-X\install"

You are now free to choose between mingw or Microsoft Visual Studios to build
the cpymad C extension.


mingw
~~~~~

For py35 or above
`````````````````

This works according to the static case (`Targetting py35 or above`_), but you
should drop all the library dependencies from the linking step (i.e. the last
command), leaving only ``-lmadx`` and the ``pythonXX.dll``.


For py34 or below
`````````````````

Just enter::

    conda install wheel cython
    python setup.py build_ext -c mingw32 --shared --madxdir=%MADXDIR%

That should be all, proceed to: Installation_.


Visual Studios
~~~~~~~~~~~~~~

Python's official binaries are all compiled with the Visual C compiler and
therefore this is the only officially supported method to build C extensions.
I will list it here for completeness.

First, look up `the correct Visual Studio version`_ and download and install
it directly from microsoft. It is possible that older versions are not
supported anymore.

.. _the correct Visual Studio version: https://wiki.python.org/moin/WindowsCompilers#Which_Microsoft_Visual_C.2B-.2B-_compiler_to_use_with_a_specific_Python_version_.3F

After that, activate the Visual Studio tools by calling ``vcvarsall.bat``.
Depending on your Visual Studio version and install path, this might look like
this::

    call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"

Finally, build cpymad::

    conda create -n py37 python=3.7
    conda activate py37
    conda install wheel cython
    python setup.py build_ext --shared --madxdir=%MADXDIR%


Installation
============

If you have arrived here, you have most of the work behind you. At this point,
you should have successfully built the python C extension.

For users
~~~~~~~~~

We now proceed to build a so called wheel_. Wheels are zip archives containing
all the files ready for installation, as well as some metadata such as version
numbers etc. The wheel can be built as follows::

    python setup.py bdist_wheel

The ``.whl`` file is named after the package and its target platform. This
file can now be used for installation on this or any other machine running the
same operating system and python version. Install as follows::

    pip install dist\cpymad-0.17.3-cp27-none-win32.whl

Finally, do a quick check that your cpymad installation is working by typing
the following::

    python -c "import cpymad.libmadx as l; l.start()"

The MAD-X startup banner should appear. Congratulations, you are now free to
delete the MAD-X and cpymad folders (but keep your wheel!).

.. _wheel: https://wheel.readthedocs.org/en/latest/


For developers
~~~~~~~~~~~~~~

If you plan on changing cpymad code, do the following instead::

    pip install -e .

Quickcheck your installation for a MAD-X startup banner by typing the
following::

    python -c "import cpymad.libmadx as l; l.start()"

You can also run more tests as follows::

    python test\test_madx.py
    python test\test_util.py
