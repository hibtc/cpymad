PyMad API
*********

Module pymad
============

The pymad package contains all parts of the general API. The abstract classes are placed
in :py:mod:`cern.pymad.abc`. In additions we have some tools which are used by both implementations,
found in :py:mod:`cern.pymad.tools`


Module cern.pymad.abc
---------------------
.. automodule:: cern.pymad.abc
    :members:


=============================
Module cern.pymad.abc.service
=============================
.. automodule:: cern.pymad.abc.service
    :members:

===========================
Module cern.pymad.abc.model
===========================
.. automodule:: cern.pymad.abc.model
    :members:



Additional tools
================

A collection of tools which can be useful to the end user are documented here.


Module cern.pymad.tfs
---------------------

Simple function which loads a tfs file::

    from cern import pymad
    table,summary=pymad.tfs('file.tfs')

You can then access e.g. the horizontal beta in several equivalent ways::

    table.betx
    table.BETX
    table['betx']
    table['BETX']

All these four methods will return the same object. Naming scheme follows the convention
from Mad-X.

It is possible to get the list of available keys::

    table.keys()
    summary.keys()
