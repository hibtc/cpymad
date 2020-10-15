.. highlight:: bash

.. _installation:

Installation
************

In order to install cpymad, please try::

    pip install cpymad --only-binary cpymad

If this fails, it usually means that we haven't uploaded wheels for your
platform or python version. In this case, either `ping us`_ about adding a
corresponding wheel, or refer to the :ref:`building-from-source` guide.

The MacOS wheels are experimental and require Apple's Accelerate framework to
be installed on the user machine. Please let us know if there are problems
with these.

Note that it is recommended to use python 3.6 or later. Support for old python
versions will be phased out when they are no longer officially supported.

Check your installation by typing the following::

    python -c "import cpymad.libmadx as l; l.start()"

The MAD-X banner should appear.

.. _ping us: https://github.com/hibtc/cpymad/issues
