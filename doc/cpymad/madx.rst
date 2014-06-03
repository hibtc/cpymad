cern.cpymad.madx
----------------

This module aims to provide the full functionality of the MAD-X library in a
more convenient Python environment. As an example, when you call
twiss/aperture/survey, you get numpy arrays containing the information from
the table generated. Further, we try to reduce the amount of commands needed
by combining e.g. use, select, and twiss into the twiss function itself, and
define reasonable default patterns/columns etc. Example:

.. code-block:: python

    from cern.cpymad.madx import Madx
    from matplotlib import pyplot as plt

    m = Madx()

    m.call('my-sequences.seq')
    m.call('my-strengths.str')

    m.command.beam(sequence='myseq1', particle='PROTON')
    # you can also just put an arbitrary MAD-X command string here:
    m.command('beam, sequence=myseq1, particle=PROTON')

    table = m.twiss('myseq1')
    columns = table.columns

    plt.plot(columns['s'], columns['betx'])
    plt.show()


Most of this should be self-explanatory.

Note, that many :class:`Madx` instances can be constructed independently
without side-effects on each other.


.. automodule:: cern.cpymad.madx
    :members:


