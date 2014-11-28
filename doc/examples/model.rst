.. _model-example:

Model example
=============

Models are information units storing all configuration of a machine, its
optics and sequences. For example:

.. code-block:: python

    from cern.cpymad import model
    from cern.resource.file import FileResource

    locator = model.Locator(FileResource('/path/to/folder/with/definitions'))
    factory = model.Factory(locator)
    model = factory('model-name')
    sequence = model.default_sequence
    twiss = sequence.twiss()

    print("max/min beta x:", max(twiss['betx']), min(twiss['betx']))
    print("ex: {0}, ey: {1}", twiss.summary['ex'], twiss.summary['ey'])

