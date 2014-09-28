.. _madx-example:

Madx example
============

Several users have been most interested in having direct access to MAD-X
inside Python instead of using models.

.. code-block:: python

    from cern.cpymad.madx import Madx
    from matplotlib import pyplot as plt

    m = Madx()

    m.call('my-sequences.seq')
    m.call('my-strengths.str')

    m.command.beam(sequence='myseq1', particle='PROTON')
    # you can also just put an arbitrary MAD-X command string here:
    m.command('beam, sequence=myseq1, particle=PROTON')

    twiss = m.twiss('myseq1')

    plt.plot(twiss['s'], twiss['betx'])
    plt.show()


Most of this should be self-explanatory.

Note, that many :class:`Madx` instances can be constructed independently
without side-effects on each other.
