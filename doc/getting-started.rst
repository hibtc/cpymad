Getting Started
~~~~~~~~~~~~~~~

The preferred way of using cpymad is via the :class:`~cpymad.madx.Madx` class.
It is responsible for controlling and accessing a MAD-X process from python.

Starting MAD-X
==============

A new MAD-X process can be spawned as follows:

.. code-block:: python

    from cpymad.madx import Madx
    madx = Madx()


Basic MAD-X commands
====================

The class has special methods to execute a few basic MAD-X commands. In
general, method names and signatures correspond closely to those of the
corresponding MAD-X commands. For example:

.. code-block:: python

    madx.call('/path/to/some/input_file.madx')

    twiss = madx.twiss(
        sequence='LEBT',
        betx=0.1, bety=0.1,
        alfx=0.1, alfy=0.1)

These methods allow short-hand notation by leaving out some parameter names
like ``file`` or ``sequence`` and sometimes do more than the MAD-X command
itself.

For example, :meth:`~cpymad.madx.Madx.call` allows to temporarily change the
directory to the one of the executed file by setting the optional ``chdir``
parameter to ``True``.

The :meth:`~cpymad.madx.Madx.twiss` method is special because it returns the
resulting twiss table as a ``dict`` like object, that can be used conveniently
for your own analysis, e.g.:

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.plot(twiss['s'], twiss['betx'])
    plt.show()

The other overloaded MAD-X commands are:

- :meth:`~cpymad.madx.Madx.survey`
- :meth:`~cpymad.madx.Madx.use`
- :meth:`~cpymad.madx.Madx.select`
- :meth:`~cpymad.madx.Madx.help`

For all other commands, you have to use one of the mechanisms described in
`Controlling MAD-X`_.


Controlling MAD-X
=================

The :class:`~cpymad.madx.Madx` class works by feeding commands in the form of
textual input to the MAD-X process. This means that you can execute all MAD-X
commands, even if they are not explicitly defined on the python class level.

input()
-------

The method responsible for feeding textual input to MAD-X is
:meth:`~cpymad.madx.Madx.input` method. It is called with a single string
argument that will be forwarded as input to the MAD-X interpreter. For
example:

.. code-block:: python

    madx.input('CALL, FILE="fodo.madx";')

Do NOT forget the trailing semicolon!

command()
---------

While it can be necessary to use :meth:`~cpymad.madx.Madx.input` for some
constructs like macros or loops, most of the time your most favorable option
is to use the :attr:`~cpymad.madx.Madx.command` attribute. It provides syntactic
sugar for composing regular MAD-X commands from python variables and feeding
the generated command string to :meth:`~cpymad.madx.Madx.input`.

.. code-block:: python

    madx.command.beam(sequence='fodo', particle='PROTON')

If you need to override how :attr:`~cpymad.madx.Madx.command` generates the
command string (argument order/formatting), you can pass strings as positional
arguments. For example:

.. code-block:: python

    madx.command.beam('sequence=fodo', particle='PROTON')

Note that positional and keyword parameters can be mixed.

A single trailing underscore will be stripped from the attribute name. This is
useful for MAD-X commands that are python keywords:

.. code-block:: python

    madx.command.global_(sequence='cassps', Q1=26.58)

In order to clone a command or element (colon syntax in MAD-X), use the
:meth:`~cpymad.madx.Command.clone` method:

.. code-block:: python

    madx.command.quadrupole.clone('QP', AT=2, L=1)

which translates to the MAD-X command::

    QP: QUADRUPOLE, AT=2, L=1;

chdir()
-------

:meth:`~cpymad.madx.Madx.chdir` changes the directory of the MAD-X process
(not the current python process).

This method is special in that it is currently the only modification of the
MAD-X interpreter state that does not go through the
:meth:`~cpymad.madx.Madx.input` method (because there is no MAD-X command to
change the directory).

Others
------

At this point, you should be able to execute arbitrary MAD-X commands via
cpymad.

All other methods for controlling MAD-X (except for
:meth:`~cpymad.madx.Madx.chdir`) are just syntactic sugar for
:meth:`~cpymad.madx.Madx.input`. Among others, this has the following main
benefits:

- every modification of the MAD-X state is transparent from the
  ``command_log`` file
- the session should be reproducible using the official ``madx`` command line
  client by the commands in the ``command_log`` file.
- reduces the need for special implementations on the cython binding by always
  going through the same interface.

More methods for changing state:

- :meth:`~cpymad.madx.Madx.verbose`: switch on or off verbose mode.


Accessing MAD-X
===============

In contrast to how cpymad is *controlling* the MAD-X state, when *accessing*
state it does not use MAD-X commands, but rather directly retrieves the data
from the C variables in the MAD-X process memory!

This means that data retrieval is relatively fast because it does **not**
require:

- a command to be parsed by the MAD-X interpreter
- to use a file on disk or the network
- output data to be parsed on the python side
- to potentially modify the MAD-X interpreter state by executing a command

Apart from this major advantage, another important implication is that the
``command_log`` file will not be cluttered by data-retrieval commands but only
show *actions*.


version
-------

Access the MAD-X version:

.. code-block:: python

    print(madx.version)
    # individual parts
    print(madx.version.date)
    print(madx.version.release)
    # or as tuple:
    print(madx.version.info >= (5, 3, 6))


elements
--------

Access to global elements:

.. code-block:: python

    # list of element names:
    print(list(madx.elements))

    # check whether an element is defined:
    print('qp1' in madx.elements)

    # get dictionary of all element properties:
    elem = madx.elements['qp1']
    print(elem['k1'])
    print(elem['l'])


tables
------

Dict-like view of MAD-X tables:

.. code-block:: python

    # list of existing table names
    print(list(madx.tables)):

    # get table as dict-like object:
    twiss = madx.tables['twiss']

    # get columns as numpy arrays:
    alfx = twiss['alfx']
    betx = twiss['betx']


variables
---------

Dictionary-like view of the MAD-X global variables:

.. code-block:: python

    # list of variable names
    print(list(madx.globals))

    # value of a builtin variable
    print(madx.globals['PI'])

Evaluate an expression in the MAD-X interpreter:

.. code-block:: python

    print(madx.evaluate('sb->angle / pi * 180'))

sequences
---------

Dictionary like view of all defined sequences:

.. code-block:: python

    # list of sequence names
    print(list(madx.sequences))

    # get a proxy object for the sequence
    fodo = madx.sequences['fodo']

    beam = fodo.beam
    print(beam['ex'], beam['ey'])

    # ordered dict-like object of explicitly defined elements:
    elements = fodo.elements

    # OR: including implicit drifts:
    expanded = fodo.expanded_elements


Logging commands
================

For the purpose of debugging, reproducibility and transparency in general, it
is important to be able to get a listing of the user input sent to
MAD-X. This can be controlled using the ``command_log`` parameter. It accepts
file names, arbitrary callables and file-like objects as follows:

.. code-block:: python

    madx = Madx(command_log="log.madx")
    madx = Madx(command_log=print)
    madx = Madx(command_log=CommandLog(sys.stderr))

Of course, in python2 the ``print`` example requires ``from __future__ import
print_function`` to be in effect.


Redirecting output
==================

The output of the MAD-X interpreter can be controlled using the ``redirect``
parameter of the :class:`~cpymad.madx.Madx` constructor. It allows to disable
the output completely:

.. code-block:: python

    madx = Madx(stdout=False)

redirect it to a file:

.. code-block:: python

    with open('madx_output.log', 'w') as f:
        madx = Madx(stdout=f)

or send the MAD-X output directly to an in-memory pipe without going through
the filesystem:

.. code-block:: python

    madx = Madx(stdout=subprocess.PIPE)
    pipe = m._process.stdout
