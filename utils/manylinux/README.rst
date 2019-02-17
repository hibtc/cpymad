manylinux
=========

This folder contains scripts for creating prebuilt wheels that can be
installed on many linux (without having to build MAD-X and cpymad manually).


Build instructions
~~~~~~~~~~~~~~~~~~

Using docker-compose
````````````````````

This requires that you have docker installed. In order to build the wheels
go to the cpymad root directory and execute::

    docker-compose -f utils/manylinux/docker-compose_x64.yml up --build

To retrieve your shiny new wheels, type::

    docker cp manylinux_cpymad_1:/io/wheels .


Using docker
````````````

If you want to (or have to) use lower level tools, you can do so as follows:

.. code-block:: bash

    docker create -v /io --name artifacts busybox

    docker build -t cpymad-manylinux utils/manylinux -f Dockerfile_x64

    docker run --init --rm \
        --volumes-from artifacts \
        -v `pwd`:/io/cpymad:ro \
        --cap-drop=all \
        cpymad-manylinux

    docker cp artifacts:/io/wheels .

Note this makes use of a data container that will persist build artifacts and
therefore significantly speedup subsequent runs.


Known issues
~~~~~~~~~~~~

Unfortunately, the available static ``libgfortran.a`` libraries (shipped in
``/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-CentOS-linux/4.8.2``) are not
compiled with -fPIC and therefore can not be used for static linking. This is
probably not a huge problem because the ``auditwheel`` step takes care to
include the ``libgfortran.so`` into the wheel.

However, the cpymad C extension still has other runtime dependencies (check
``readelf -d``) that are not included into the wheel. I am not sure whether
any of these may be problematic:

- ``libstdc++.so.6``
- ``libgcc_s.so.1``
- ``libpthread.so.0``
- ``libc.so.6``
