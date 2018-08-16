manylinux
=========

This folder contains scripts for creating prebuilt wheels that can be
installed on many linux (without having to build MAD-X and cpymad manually).


Build instructions
~~~~~~~~~~~~~~~~~~

It is possible to build cpymad wheels with a MAD-X version from the working
directory by cloning it here and modifying as desired::

    git clone git@github.com:MethodicalAcceleratorDesign/MAD-X

This requires that you have docker installed. The wheels are built as
follows::

    docker-compose up

Or manually:

.. code-block:: bash

    docker build -t cpymad-manylinux .

    # if running from current directory:
    docker run --rm --init \
        -v `pwd`:/io \
        -v `pwd`/../..:/io/cpymad \
        --cap-drop=all \
        cpymad-manylinux

    # OR, running from cpymad root directory:
    docker run --rm --init \
        -v `pwd`/utils/manylinux:/io \
        -v `pwd`:/io/cpymad \
        --cap-drop=all \
        cpymad-manylinux

When debugging, it can be useful to run an interactive session in the docker
container::

    docker run --rm -it cpymad-manylinux

Finally, find and upload your wheels in the ``wheels/`` subdirectory::

    twine upload wheels/*.whl


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
