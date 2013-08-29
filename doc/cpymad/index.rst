CPyMad implementation
*********************

CPyMad is designed for two use-cases.

* The first is as a replacement for
  the Mad-X interpreter, where function calls are pretty much the same, but
  are converted to Python-like syntax. The module is called :py:class:`cern.madx`.
* The second is a model object, which is built on top of the new Mad-X interface.
  The model object is aimed to be using model definitions from JMad, where all the
  preliminary setup for a given machine is defined in built-in files, including knowledge
  about available optics, strength files to use for each etc. The module is called
  :py:class:`cern.cpymad.model`

Module cern.madx
----------------

This module aims to provide the full functionality of the Mad-X library in a more
convenient Python environment. As an example, when you call twiss/aperture/survey etc,
you immediately retrieve a functional Python object containing the information from the
table generated. Further, we try to reduce the amount of commands needed by combining
e.g. use, select, and twiss into the twiss function itself, and define reasonable default
patterns/columns etc.

A simple use case::

    from cern import madx
    from matplotlib import pyplot

    m=madx.madx()

    m.call('my-sequences.seq')
    m.call('my-strengths.str')

    m.command('beam,sequence=myseq1,particle=PROTON')

    tw,summary=m.twiss('myseq1',columns=['s','betx'])

    pyplot.plot(tw.s,tw.betx)
    pyplot.show()

In this example we assume that you have a file containing the Mad-X commands to set up the sequence
and the strengths for your optics. You can then use :py:meth:`~cern.madx.madx.call` for which
the syntax should be understandable for a Mad-X user.

For anything not yet implemented in :py:class:`cern.madx.madx`,
the :py:meth:`~cern.madx.madx.command` function is there. You can give any normal
Mad-X like command to this function. If you forget the **;** at the end, it will be
automatically added. In the example it is used to set the beam.

Finally we call twiss. If we do not give a pattern, 'full' is chosen by default. Columns are also optional,
but it is recommended to set them for speed improvement. Currently the table is first written to file and then
parsed into a dictionary-like Python object. Each column in the table containing numerical data are in the form
of numpy.array() objects, while strings are simple lists.

Finally we choose another library to plot the result, in this case :py:meth:`matplotlib.pyplot.plot`.

Automatic documentation
=======================


.. automodule:: cern.madx
    :members:


Module cern.cpymad.model
------------------------

This module is an implementation of the abstract class
:py:class:`cern.pymad.abc.model`, using :py:class:`cern.madx` as the backend.

A simple use-case::

    from cern import cpymad
    m=cpymad.model('lhc')
    tw=m.twiss()[0]
    print "max/min beta x:",max(tw.betx),min(tw.betx)

Automatic documentation
=======================

.. automodule:: cern.cpymad.model
    :members:
