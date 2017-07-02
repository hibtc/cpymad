Usage
~~~~~

The ``Madx`` class provides a basic binding to the MAD-X interpreter:

.. code-block:: python

    from cpymad.madx import Madx

    # Start a MAD-X interpretor. All MAD-X commands issued via cpymad will
    # be logged to `command_log`:
    madx = Madx(command_log="log.madx")

    # Show the version of MAD-X that is actually loaded:
    print(madx.version)

    # Execute one of your predefined MAD-X files:
    madx.call('/path/to/some/input_file.madx')

    # Only a handful of MAD-X methods are exposed as methods. For others,
    # you can use the `command` attribute. For example, to set a beam:
    madx.command.beam(sequence='myseq1', particle='PROTON')

    # Calculate TWISS parameters:
    twiss = madx.twiss(sequence='LEBT',
                       betx=0.1, bety=0.1,
                       alfx=0.1, alfy=0.1)

    # Your own analysis below:
    from matplotlib import pyplot as plt
    plt.plot(twiss['s'], twiss['betx'])
    plt.show()

There are alternative syntaxes for the extreme cases where you need more
fine grained control over the command string composition or where
``command`` fails to do the right thing:

.. code-block:: python

    # can't use `global` as attribute, since it's a python keyword:
    madx.command.global_(sequence='cassps', Q1=26.58)

    # can't use `: ` as attribute:
    madx.command('QP: QUADRUPOLE', AT=2, L=1)

    # issue a plain text command, don't forget the semicolon!
    madx.input('FOO, BAR=[baz], QUX=<NORF>;')
