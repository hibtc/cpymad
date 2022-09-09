.. highlight:: bash

Linux
-----

cpymad is linked against a library version of MAD-X, which means that in order
to build cpymad you first have to compile MAD-X from source. The official
``madx`` executable is not sufficient. These steps are described in the
following subsections:

.. contents:: :local:


Build MAD-X
~~~~~~~~~~~

In order to build MAD-X from source, please install the following build tools:

- CMake_ >= 3.0
- gcc >= 4.8
- gfortran

Other C/C++/fortran compiler suites may work too but are untested as of now.

Download and extract the latest `MAD-X release`_ from github, e.g.:

.. code-block:: bash
    :substitutions:

    wget https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/|VERSION|.tar.gz
    tar -xzf MAD-X-|VERSION|.tar.gz

.. _CMake: http://www.cmake.org/
.. _MAD-X release: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases

or directly checkout the source code using git (unstable)::

    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X

We will do an out-of-source build in a ``build/`` subdirectory. This way, you
can easily delete the ``build`` directory and restart if anything goes wrong.
The basic process looks as follows::

    mkdir MAD-X/build
    cd MAD-X/build

    cmake .. \
        -DMADX_ONLINE=OFF \
        -DMADX_INSTALL_DOC=OFF \
        -DCMAKE_INSTALL_PREFIX=../dist \
        -DCMAKE_C_FLAGS="-fvisibility=hidden"

    make install

Here we have specified a custom installation prefix to prevent cmake from
installing MAD-X to a system directory (which would require root privileges,
and may be harder to remove completely). You can also set a more permanent
install location if you prefer (e.g. ``~/.local`` or ``/opt/madx``), but keep
in mind that there is no ``uninstall`` command other than removing the files
manually.

The cmake command has many more options, the most important ones being
(only use if you now what you're doing!):

- ``-DMADX_STATIC=ON``: Pass this flag to link statically against the
  dependencies of MAD-X (libc, libgfortran, libstdc++, blas, lapack, etc).
  This may be attempted in case of problems and is not guaranteed to work on
  all platforms (if your OS e.g. does not distribute ``libgfortran.a`` as is
  the case on archlinux). Note that even without this flag, cpymad will still
  be linked statically against MAD-X, just not against its dependencies.

- ``-DBUILD_SHARED_LIBS=ON``: Pass this flag if you want to link cpymad
  dynamically against MAD-X. In theory, this allows using, testing and even
  updating the MAD-X shared object independently of cpymad. If using this
  option, also change ``-DCMAKE_C_FLAGS="-fvisibility=protected"`` and be
  aware that you have to redistribute the MAD-X shared object along with
  cpymad, or install MAD-X to a permanent location where it can be found at
  runtime. Usually this means installing to the (default) system directories,
  but it can also be done by setting the LD_LIBRARY_PATH_ environment variable
  or passing appropriate ``--rpath`` to the setup script.

.. _LD_LIBRARY_PATH: http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html

Save the path to the install directory in the ``MADXDIR`` environment variable.
This variable will be used later by the ``setup.py`` script to locate the
MAD-X headers and library, for example::

    export MADXDIR="$(pwd)"/../dist

Also, set the following variables according to the flags passed to the cmake
command above (ONLY PASS IF NEEDED!)::

    set STATIC=1    # if -DMADX_STATIC=ON
    set SHARED=1    # if -DBUILD_SHARED_LIBS=ON


Building cpymad
~~~~~~~~~~~~~~~

Install setup requirements::

    pip install cython wheel

Enter the cpymad folder, and build as follows::

    python setup.py build_ext -lm

The ``-lm`` might not be necessary on all systems.

If you have installed blas/lapack and MAD-X found it during the cmake step,
you have to pass them as additional link libraries::

    python setup.py build_ext -lm -lblas -llapack

You can now create and install a wheel as follows (however, note that this
wheel probably won't be fit to be distributed to other systems)::

    python setup.py bdist_wheel
    pip install dist/cpymad-*.whl

If you plan on changing cpymad code, do the following instead::

    pip install -e .
