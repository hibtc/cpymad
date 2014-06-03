API Reference
*************

There are two main components in CPyMAD:

* :class:`cern.madx.Madx` serves as a lightweight wrapper for the MAD-X
  interpretor.

* :class:`cern.cpymad.model.Model` is derived from the JMad model
  definitions. It stores all preliminary setup for a given machine in
  built-in files, including knowledge about available optics and strength
  files to use for each, etc.


.. toctree::
   :maxdepth: 2

   madx
   model
