.. _model-example:

Model example
=============

Models are information units storing all configuration of a machine, its
optics and sequences. For example:

.. code-block:: python

    from cern.cpymad.model import Model

    model = Model.load('/path/to/model/definition.cpymad.yml')

    twiss = model.default_sequence.twiss()

    print("max/min beta x:", max(twiss['betx']), min(twiss['betx']))
    print("ex: {0}, ey: {1}", twiss.summary['ex'], twiss.summary['ey'])

