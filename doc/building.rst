.. _building-from-source:

Building from source
********************

The following sections contain instructions for building MAD-X and cpymad from
source on various platforms. Ordinary users usually don't need to do this and
can should the prebuilt wheels instead (see :ref:`installation`). Good reasons
to build cpymad from source are, e.g.:

- you are a package maintainer
- you want to change the cpymad code
- you want to build cpymad against a specific version of MAD-X
- your target version of python and/or platform is unsupported and we don't
  want to start supporting it for some reason

Note that cpymad is linked against a library version of MAD-X, which means
that in order to build cpymad you first have to compile MAD-X from source
(even if you have the MAD-X executable installed).

.. rubric:: Contents

.. toctree::
   :maxdepth: 1

   installation/unix
   installation/windows
   installation/troubleshooting
