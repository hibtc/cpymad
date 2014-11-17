.. _model-example:

Model example
=============

Models are information units storing all configuration of a machine, its
optics and sequences. For example:

.. code-block:: python

    from cern.cpymad.api import load_model

    model = load_model('lhc')
    twiss, summary = model.twiss()
    print("max/min beta x:", max(twiss['betx']), min(twiss['betx']))

