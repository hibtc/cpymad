cern.cpymad.model
-----------------

This module provides a model class which is conceptually derived from the
jmad models, but uses :class:`cern.cpymad.madx.Madx` as the backend.
Example:

.. code-block:: python

    from cern.cpymad import load_model

    m = load_model('lhc')
    twiss, summary = m.twiss()
    print("max/min beta x:", max(twiss['betx']), min(twiss['betx']))


.. automodule:: cern.cpymad.model
    :members:
