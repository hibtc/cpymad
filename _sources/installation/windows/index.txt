Windows
-------

On a windows platform, you are likely to run into more problems than on
linux when building from source. Therefore, I will try to provide `built
versions`_ for some platforms. If your platform is supported, you can just
run

.. code-block:: bat

    pip install cpymad

from your terminal and that's it. If you want to install to a target
machine without internet access (or with firewall), you can manually
download the ``.whl`` file for your platform and then use pip_ to install
that particular file. For example:

.. code-block:: bat

    pip install dist\cpymad-0.10.1-cp27-none-win32.whl

In this case you will also need to grab and install *numpy*, *PyYAML* and
*setuptools* (and possibly their dependencies if any). Installable wheel_
archives for these packages can be conveniently downloaded from here:
`Unofficial Windows Binaries for Python Extension Packages`_.

If there is no binary available for your platform you can ask me via email
or the github issue pages to create one for your python version / platform.

If you want to build from source, have a look at the following guides:

.. toctree::
   :maxdepth: 1

   pythonxy
   winpython
   other

I recommend WinPython, which allows to create self-contained binaries that
can later be used on other python distributions as well. Furthermore, being
portable, WinPython does not change any system settings during its
installation (a.k.a. extraction).


.. _built versions: https://pypi.python.org/pypi/cpymad/#downloads
.. _pip: https://pypi.python.org/pypi/pip
.. _wheel: https://wheel.readthedocs.org/en/latest/
.. _Unofficial Windows Binaries for Python Extension Packages: http://www.lfd.uci.edu/~gohlke/pythonlibs/
