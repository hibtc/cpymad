manylinux
=========

This folder contains scripts for creating prebuilt wheels that can be
installed on many linux (without having to build MAD-X and cpymad manually).


Build instructions
~~~~~~~~~~~~~~~~~~

This requires that you have docker running. In order to build the wheels
go to the cpymad root directory and execute::

    docker create -v /io --name artifacts busybox

    docker run --init --rm \
        --volumes-from artifacts \
        -v `pwd`:/io/cpymad:ro \
        -v /io/cpymad/src/cpymad.egg-info \
        --cap-drop=all \
        quay.io/pypa/manylinux2010_x86_64 \
        /io/cpymad/utils/manylinux/build_all

    docker cp artifacts:/io/dist .

This makes use of a data container that will persist build artifacts and
therefore significantly speedup subsequent runs.

Note the use of a separate volume for the ``.egg-info`` directory! This is
needed to allow mounting all host filesystems as read-only — which avoids a
number of potential permission issues in both host and container.


Known issues
~~~~~~~~~~~~

Unfortunately, the available static ``libgfortran.a`` libraries (shipped in
``/opt/rh/devtoolset-2/root/usr/lib/gcc/x86_64-CentOS-linux/4.8.2``) are not
compiled with -fPIC and therefore can not be used for static linking. This is
probably not a huge problem because the ``auditwheel`` step takes care to
include the ``libgfortran.so`` into the wheel.
