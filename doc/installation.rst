.. highlight:: bash

.. _installation:

Installation
************

In order to install cpymad, please try::

    pip install cpymad --only-binary cpymad

If this fails, it usually means that we haven't uploaded wheels for your
platform or python version. In this case, either `ping us`_ about adding a
corresponding wheel, or refer to the :ref:`building-from-source` guide.

Note that it is recommended to use python 3.6 or later. Support for old python
versions will be phased out when they are no longer officially supported.

.. _ping us: https://github.com/hibtc/cpymad/issues


Offline installation
====================

In order to install to a target machine without internet access (or behind a
nasty firewall) ``pip`` can be used to download all required files in advance::

    pip download -d wheels cpymad

and then later install them in the offline environment like this::

    pip install -f wheels cpymad

Optionally, pass ``--platform <platform> --python-version <version>`` to the
``pip download`` command if downloading wheels on a platform that is different
from the target system.

Individual wheels can also be downloaded manually from PyPI_ and later
installed using pip, e.g.::

    pip install cpymad-1.6.1-cp37-cp37m-win_amd64.whl

In this case, you should ensure to also have all dependencies available.

.. _PyPI: https://pypi.python.org/pypi/cpymad/#downloads
