.. highlight:: bash

MacOS (experimental)
--------------------

cpymad is linked against a library version of MAD-X, which means that in order
to build cpymad you first have to compile MAD-X from source. The official
``madx`` executable is not sufficient. These steps are described in the
following subsections:

.. contents:: :local:


Build MAD-X
~~~~~~~~~~~

In order to build MAD-X from source, please install the following build tools:

- gcc >= 4.8
- gfortran
- CMake_ >= 3.0 (e.g. using ``pip install cmake``)

Other C/C++/fortran compiler suites may work too but are untested as of now.

Download and extract the latest `MAD-X release`_ from github, e.g.:

.. code-block:: bash
    :substitutions:

    curl -L -O https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/|VERSION|.tar.gz
    tar -xzf |VERSION|.tar.gz

.. _CMake: http://www.cmake.org/
.. _MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases

or directly checkout the source code using git (unstable)::

    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X

On Mac, you currently also have to apply the following patch_ to the MAD-X
source, to make the build work::

    curl -L -O https://raw.githubusercontent.com/hibtc/cpymad/master/.github/patch/fix-macos-symbol-not-found-mad_argc.patch
    patch -d ./MAD-X -p1 <fix-macos-symbol-not-found-mad_argc.patch

.. _patch: https://raw.githubusercontent.com/hibtc/cpymad/master/.github/patch/fix-macos-symbol-not-found-mad_argc.patch

We will do an out-of-source build in a ``build/`` subdirectory. This way, you
can easily delete the ``build`` directory and restart if anything goes wrong.
The basic process looks as follows::

    pip install --upgrade cmake

    mkdir MAD-X/build
    cd MAD-X/build

    cmake .. \
        -DCMAKE_POLICY_DEFAULT_CMP0077=NEW \
        -DCMAKE_POLICY_DEFAULT_CMP0042=NEW \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_Fortran_COMPILER=gfortran \
        -DBUILD_SHARED_LIBS=OFF \
        -DMADX_STATIC=OFF \
        -DCMAKE_INSTALL_PREFIX=../dist \
        -DCMAKE_BUILD_TYPE=Release \
        -DMADX_INSTALL_DOC=OFF \
        -DMADX_ONLINE=OFF \
        -DMADX_FORCE_32=OFF \
        -DMADX_X11=OFF
    cmake --build . --target install

Here we have specified a custom installation prefix to prevent cmake from
installing MAD-X to a system directory (which would require root privileges,
and may be harder to remove completely). You can also set a more permanent
install location if you prefer, but keep in mind that there is no
``uninstall`` command other than removing the files manually.

The cmake command has many more options, but these are untested on Mac so far.

Save the path to the install directory in the ``MADXDIR`` environment variable.
This variable will be used later by the ``setup.py`` script to locate the
MAD-X headers and library, for example::

    export MADXDIR="$(pwd)"/../dist


Building cpymad
~~~~~~~~~~~~~~~

Install setup requirements::

    pip install -U cython wheel setuptools delocate

Enter the cpymad folder, and build as follows::

    export CC=gcc

    python setup.py build_ext

If you have installed blas/lapack and MAD-X found it during the cmake step,
you have to pass them as additional link libraries::

    python setup.py build_ext -lblas -llapack

You can now create and install a wheel as follows (however, note that this
wheel probably won't be fit to be distributed to other systems)::

    python setup.py bdist_wheel
    delocate-wheel dist/*.whl
    pip install dist/cpymad-*.whl

If you plan on changing cpymad code, do the following instead::

    pip install -e .
