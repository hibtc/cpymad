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

.. _built versions: https://pypi.python.org/pypi/cpymad/#downloads


Manual build
============

In order to build cpymad manually, I recommend using **conda** which makes it
very easy to install different python versions, quickly setup and tear down
independent environments and even acquire and build non-python dependencies
and compilers! Miniconda_ is perfectly suitable for all our intents, but
anaconda should work fine too.

.. _miniconda: https://conda.io/en/latest/miniconda.html

Build libmadx
~~~~~~~~~~~~~

After installing conda, open the conda command prompt. In order to build
MAD-X, first create a python 3.4 environment with cmake_ and mingwpy_ -- even
if you plan using cpymad on a higher python version! If you want cpymad on
py2.7 or 3.3 use the target python version for building MAD-X instead of 3.4.

.. code-block:: bat

    conda create -n py34 python=3.4 cmake
    conda activate py34
    conda install -c conda-forge mingwpy

.. _cmake: http://www.cmake.org/
.. _mingwpy: https://mingwpy.github.io/

Now download and extract the `latest MAD-X release`_.

.. _latest MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases

I recommend to build MAD-X as a *static* library as described below. This
way, you won't need to carry any ``.dll`` files around and you won't run
into version problems when having a multiple MAD-X library builds around.

However, it is somewhat easier to link cpymad *dynamically* against MAD-X on
python versions above 3.4, i.e. using a DLL. This comes at the cost of having
to redistribute the ``madx.dll`` along. The choice is yours.

Static build:

.. code-block:: bat

    mkdir MAD-X\build-static
    cd MAD-X\build-static
    cmake .. ^
        -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=OFF ^
        -DMADX_STATIC=ON ^
        -DMADX_INSTALL_DOC=OFF ^
        -DCMAKE_INSTALL_PREFIX=..\install

DLL build:

.. code-block:: bat

    mkdir MAD-X\build-shared
    cd MAD-X\build-shared
    cmake .. ^
        -G "MinGW Makefiles" ^
        -DBUILD_SHARED_LIBS=ON ^
        -DMADX_STATIC=OFF ^
        -DMADX_INSTALL_DOC=OFF ^
        -DCMAKE_INSTALL_PREFIX=..\install

The following step will build the library. This may take a few minutes, so go
and grab a coffee meanwhile:

.. code-block:: bat

    mingw32-make
    mingw32-make install

If all went well the last command will have installed binaries and library
files to the :file:`%MADX%\\install` subfolder.

Save the full path to the install directory in the ``MADXDIR`` enviroment
variable, this variable will be used later by the ``setup.py`` script to
locate the MAD-X headers and library:

.. code-block:: bat

    set "MADXDIR=C:\Users\<....>\MAD-X\install"


Get cpymad source
~~~~~~~~~~~~~~~~~

Next, download and extract the latest `cpymad release`_. Alternatively, use
git to retrieve the current development version (unstable):

.. code-block:: bat

    git clone https://github.com/hibtc/cpymad


Build cpymad on py34 or below
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to the cpymad folder and build a so called wheel_, which is essentially a
zip archive containing all the files ready for installation:

.. code-block:: bat

    conda install wheel cython
    python setup.py build_ext -c mingw32 --static --madxdir=../MAD-X/install
    python setup.py bdist_wheel

If you built MAD-X as DLL (dynamic build), just replace ``--static`` in the
second line by ``--shared``.


Build cpymad on py35 or above
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an environment with your target python version, e.g.:

.. code-block:: bat

    conda create -n py37 python=3.7 wheel cython
    conda activate py37

If you created MAD-X as DLL, the build should work essentially the same as
on earlier python versions, except that you have to install the Visual C
compiler first and activate the compiler environment as follows:

.. code-block:: bat

    call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"

Then

.. code-block:: bat

    python setup.py build_ext --shared --madxdir=<madx-install-path>
    python setup.py bdist_wheel

Now comes the tricky part, if you have built MAD-X as a static library, you
will have to "cross-compile" (sort of) the cython extension on the target
platform with GCC from mingwpy in python 3.4. First, set a few environment
variables with the path of GCC, the python prefix of the target python version
and certain platform/abi tags:

.. code-block:: bat

    for /f %G in ('python -c "import sys; print(sys.prefix)"') do (
        set "gcc=%~fG\..\py34\Scripts\gcc.exe"
    )

    for /f %G in ('python -c "import sys; print(sys.prefix)"') do (
        set "pythondir=%~fG"
    )

    set py_ver=37
    set dir_tag=win-amd64-3.7
    set file_tag=cp37-win_amd64

    set tempdir=build\temp.%dir_tag%\Release\src\cpymad
    set libdir=build\lib.%dir_tag%\cpymad

And use this for good as follows:

.. code-block:: bat

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

    python setup.py build_wheel


Install cpymad
~~~~~~~~~~~~~~

The ``.whl`` file is named after the package and its target platform. This
file can now be used for installation like so:

.. code-block:: bat

    pip install dist\cpymad-0.17.3-cp27-none-win32.whl


.. _cpymad release: https://github.com/hibtc/cpymad/releases
.. _wheel: https://wheel.readthedocs.org/en/latest/
