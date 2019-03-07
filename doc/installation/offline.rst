.. highlight:: bat

Offline installation
====================

If you want to install to a target machine without internet access (or with
firewall), you can manually download_ the ``.whl`` file for your platform and
then use pip_ to install that particular file. For example::

    pip install cpymad-0.17.3-cp27-none-win32.whl

In this case you will also need to grab and install *numpy* and
*setuptools* (and possibly their dependencies if any). Installable wheel_
archives for these packages can be conveniently downloaded from here:
`Unofficial Windows Binaries for Python Extension Packages`_.

To simplify this process, ``pip`` can automatically download all required
files::

    mkdir packages
    pip download -d packages cpymad

and then later install them in the offline environment like this::

    pip install -f packages cpymad

If you want/have to build from source, have a look at following guide.

.. _download: https://pypi.python.org/pypi/cpymad/#downloads
.. _pip: https://pypi.python.org/pypi/pip
.. _wheel: https://wheel.readthedocs.org/en/latest/
.. _Unofficial Windows Binaries for Python Extension Packages: http://www.lfd.uci.edu/~gohlke/pythonlibs/
