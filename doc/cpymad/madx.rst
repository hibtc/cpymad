cern.cpymad.madx
----------------

This module aims to provide the full functionality of the MAD-X library in a
more convenient Python environment. For example, when you call the ``twiss``
method, you get numpy arrays containing the information from the table
generated. Furthermore, we try to reduce the amount of commands needed by
combining e.g. USE, SELECT, and TWISS into the ``twiss`` method itself, and
define reasonable default patterns/columns etc. See, :ref:`madx-example`.

.. automodule:: cern.cpymad.madx
    :members:
