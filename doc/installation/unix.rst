Unix
----
There are now binary wheels for most supported target platforms. With these
the installation is as simple as:

.. code-block:: bash

    pip install cpymad

If a wheel for your platform is not available, this command may pick up the
source package and try to build from scratch.

Since cpymad is linked against a special library version of MAD-X, and
therefore must usually compile MAD-X from source before it can be built (even
if you have the MAD-X executable installed). This means that the cpymad setup
can take between 5 and 30 minutes, depending on your internet bandwidth and
machine performance.

In case of failure, you should build MAD-X and cpymad manually, as described
below.


Install libmadx
~~~~~~~~~~~~~~~

cpymad requires a library build of MAD-X. The official executable is not
sufficient. In order to build MAD-X from source you will need the following
programs:

- CMake_ >= 3.0
- gcc >= 4.8, along with gfortran (other C/C++/fortran compiler suites may
  work too but are untested as of now)

Download the `latest MAD-X release`_ `from github`_:

.. code-block:: bash

    wget https://github.com/MethodicalAcceleratorDesign/MAD-X/archive/5.04.02.tar.gz
    tar -xzf MAD-X-5.04.02.tar.gz

or use directly the source code from master (unstable):

.. code-block:: bash

    git clone https://github.com/MethodicalAcceleratorDesign/MAD-X

We will do an out-of-source build in a ``build/`` subdirectory. This way, you
can easily delete the ``build`` directory and restart if anything goes wrong.
The basic process looks as follows:

.. code-block:: bash

    cd MAD-X
    mkdir build && cd build
    cmake .. \
        -DMADX_ONLINE=OFF \
        -DMADX_INSTALL_DOC=OFF \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_C_FLAGS="-fvisibility=hidden"
    make && make install

Here, we have turned off the online model, documentation files and, more
importantly, have used a custom installation prefix to prevent cmake from
installing MAD-X to a system directory (which would require root privileges,
and may be harder to remove completely). The last argument prevents a class of
crashes due to symbol collisions with other subsystems such as the C standard
library.

If you prefer a more permanent install location
(``-DCMAKE_INSTALL_PREFIX=XXX``), the most common ones are as follows::

    ~/.local                user-level installation, no sudo required
    /opt/madx               system-wide but easily removable installation
    /opt/madx/<VERSION>     if you plan to install several versions side-by-side
    /usr                    system, usually default, mixes with your system files
    /usr/local              system, local machine (not always used), mixes like /usr

Although the cmake command has many more options I will mention only the
following:

You can pass ``-DMADX_STATIC=ON`` to specify that the link dependencies of
MAD-X itself (libc, libgfortran, libstdc++, blas, lapack, etc) should be
linked statically when building cpymad later. This may be attempted in case of
problems and is not guaranteed to work on all platforms (if your OS e.g.  does
not distribute ``libgfortran.a`` as is the case on archlinux). Note that even
if you do not set this flag, cpymad will still be linked statically against
MAD-X, just not against the c/c++/fortran/etc runtimes.

You can pass ``-DBUILD_SHARED_LIBS=ON`` if you want to link cpymad dynamically
against MAD-X. In theory, this allows using, testing and even updating the
MAD-X shared object independently of cpymad, but probably does more harm than
good in practice. If using this option, please change the visibility to
``-DCMAKE_C_FLAGS="-fvisibility=protected"`` and be aware that you have to
make sure to **install MAD-X to a permanent location** where it can be found
at runtime. Usually this means installing to the (default) system directories,
but it can also be done by setting the LD_LIBRARY_PATH_ environment variable
or passing appropriate ``--rpath`` to the setup script.

.. _CMake: http://www.cmake.org/
.. _latest MAD-X release: http://madx.web.cern.ch/madx/releases/last-rel
.. _from github: https://github.com/MethodicalAcceleratorDesign/MAD-X/releases
.. _LD_LIBRARY_PATH: http://tldp.org/HOWTO/Program-Library-HOWTO/shared-libraries.html


Install cpymad
~~~~~~~~~~~~~~

After having built MAD-X we can now build cpymad. You will need:

- python >= 2.7, but higher versions (>=3.7) are preferred
- setuptools_ python package
- Cython_, if you plan to work with the git checkout. Cython can be installed
  using ``pip install cython`` (this is unnecessary for the release tarball
  from PyPI)

Concerning runtime dependencies on other python packages, cpymad requires only
numpy_ and minrpc_, both of which should usually be resolved automatically by
pip_ or the setup script. If you plan to install in an offline environment,
you can download all dependencies using the command ``pip download cpymad``.

.. _setuptools: https://pypi.org/project/setuptools
.. _cython:     http://cython.org/
.. _numpy:      http://www.numpy.org/
.. _pip:        https://pypi.org/project/pip
.. _minrpc:     https://pypi.org/project/minrpc

We will need to tell the cpymad setup script to use our MAD-X installation
path from before. The easiest way to do this is by setting an environment
variable:

.. code-block:: bash

    export MADXDIR=/PATH/TO/CMAKE_INSTALL_PREFIX

If you did build MAD-X with ``-DBUILD_SHARED_LIBS`` or ``-DMADX_STATIC``
you should also set the corresponding option:

.. code-block:: bash

    export SHARED=1

    # or:

    export STATIC=1

With these settings in place, you can try installing cpymad as before:

.. code-block:: bash

    pip install --no-binary=cpymad cpymad


Building cpymad manually
~~~~~~~~~~~~~~~~~~~~~~~~

If the installation fails or produces an unloadable version of cpymad, fetch
`latest cpymad release`_ from PyPI (the idea is that this grants you more
control over the build options and alter the setup script if necessary):

.. code-block:: bash

    pip download --no-binary=cpymad --no-deps cpymad
    tar -xzf cpymad-*.tar.gz

Alternatively, fetch the very latest cpymad_ source_ from git:

.. code-block:: bash

    git clone https://github.com/hibtc/cpymad

After that, build cpymad and enter development mode so that changes in the
local directory will take effect immediately (don't forget to export the MAD-X
path as above):

.. code-block:: bash

    cd cpymad
    python setup.py build_ext

The advantage with this method is that you can pass additional compiler or
linker arguments to the ``build_ext`` command. For example, if you happened to
build MAD-X with blas/lapack, you may need to pass additional linklibs:

.. code-block:: bash

    python setup.py build_ext -lblas -llapack

Once you get cpymad working you may wish to make your installation more
permanent, by e.g. using the ``install`` command:

.. code-block:: bash

    python setup.py install

Or even creating a wheel that can be installed using pip:

.. code-block:: bash

    python setup.py bdist_wheel
    pip install dist/cpymad-*.whl


.. _latest cpymad release: https://pypi.org/project/cpymad#files
.. _pip: https://pypi.org/project/pip
.. _cpymad: https://github.com/hibtc/cpymad
.. _source: https://github.com/hibtc/cpymad/zipball/master
