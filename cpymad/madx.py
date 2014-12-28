"""
This module defines a convenience layer to access the MAD-X interpreter.

The most interesting class for users is :class:`Madx`.
"""

from __future__ import absolute_import

from functools import partial
import logging
import os
import collections

from . import _rpc
from . import util

try:
    basestring
except NameError:
    basestring = str


class Version(object):

    """Version information struct. """

    def __init__(self, release, date):
        """Store version information."""
        self.release = release
        self.date = date

    def __repr__(self):
        """Show nice version string to user."""
        return "MAD-X {} ({})".format(self.release, self.date)


class ChangeDirectory(object):

    """Context manager for temporarily changing current working directory."""

    def __init__(self, path, _os=os):
        """
        Change the path using the given os module.

        :param str path: new path name
        :param module _os: module with 'getcwd' and 'chdir' functions
        """
        self._os = _os
        # Contrary to common implementations of a similar context manager,
        # we change the path immediately in the constructor. That enables
        # this utility to be used without any 'with' statement:
        if path:
            self._restore = _os.getcwd()
            _os.chdir(path)
        else:
            self._restore = None

    def __enter__(self):
        """Enter 'with' context."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit 'with' context and restore old path."""
        if self._restore:
            self._os.chdir(self._restore)


class MadxCommands(object):

    """
    Generic MAD-X command wrapper.

    Raw python interface to issue MAD-X commands. Usage example:

    >>> command = MadxCommands(libmadx.input)
    >>> command('twiss', sequence='LEBT')
    >>> command.title('A meaningful phrase')

    :ivar __dispatch: callable that takes a MAD-X command string.
    """

    def __init__(self, dispatch):
        """Set :ivar:`__dispatch` from :var:`dispatch`."""
        self.__dispatch = dispatch

    def __call__(self, *args, **kwargs):
        """Create and dispatch a MAD-X command string."""
        self.__dispatch(util.mad_command(*args, **kwargs))

    def __getattr__(self, name):
        """Return a dispatcher for a specific command."""
        return partial(self.__call__, name)


def NOP(s):
    """Do nothing."""
    pass


class CommandLog(object):

    """Log MAD-X command history to a file text."""

    @classmethod
    def create(cls, filename):
        """Create CommandLog from filename (overwrite/create)."""
        return cls(open(filename, 'wt'))

    def __init__(self, file):
        """Create CommandLog from file instance."""
        self._file = file

    def __call__(self, command):
        """Log a single history line and flush to file immediately."""
        self._file.write(command + '\n')
        self._file.flush()


class Madx(object):

    """
    Each instance controls one MAD-X process.

    This class aims to expose a pythonic interface to the full functionality
    of the MAD-X library. For example, when you call the ``twiss`` method, you
    get numpy arrays containing the information from the table generated.
    Furthermore, we try to reduce the amount of commands needed by combining
    e.g. USE, SELECT, and TWISS into the ``twiss`` method itself, and define
    reasonable default patterns/columns.

    The following very simple example demonstrates basic usage:

    .. code-block:: python

        from cpymad.madx import Madx

        m = Madx()

        m.call('my-sequences.seq')
        m.call('my-strengths.str')

        m.command.beam(sequence='myseq1', particle='PROTON')
        # you can also just put an arbitrary MAD-X command string here:
        m.command('beam, sequence=myseq1, particle=PROTON')

        twiss = m.twiss('myseq1')

        # now do your own analysis:
        from matplotlib import pyplot as plt
        plt.plot(twiss['s'], twiss['betx'])
        plt.show()

    By default :class:`Madx` uses a subprocess to execute MAD-X library calls
    remotely via a simple RPC protocol which is defined in :mod:`_rpc`. If
    required this behaviour can be customized by passing a custom ``libmadx``
    object to the constructor. This object must expose an interface similar to
    :mod:`libmadx`.
    """

    def __init__(self, libmadx=None, command_log=None, error_log=None):
        """
        Initialize instance variables.

        :param libmadx: :mod:`libmadx` compatible object
        :param command_log: logs MAD-X history either filename or CommandLog
        :param error_log: logger instance ``logging.Logger``
        """
        # get logger
        if error_log is None:
            error_log = logging.getLogger(__name__)
        # open history file
        if isinstance(command_log, basestring):
            command_log = CommandLog.create(command_log)
        elif command_log is None:
            command_log = NOP
        # start libmadx subprocess
        if libmadx is None:
            svc, proc = _rpc.LibMadxClient.spawn_subprocess()
            libmadx = svc.libmadx
        if not libmadx.started():
            libmadx.start()
        # init instance variables:
        self._libmadx = libmadx
        self._command_log = command_log
        self._error_log = error_log

    @property
    def version(self):
        """Get the MAD-X version."""
        return Version(self._libmadx.version(),
                       self._libmadx.release_date())

    @property
    def command(self):
        """
        Perform a single MAD-X command.

        :param str cmd: command name
        :param **kwargs: command parameters
        """
        return MadxCommands(self.input)

    def input(self, text):
        """
        Run any textual MAD-X input.

        :param str text: command text
        """
        # write to history before performing the input, so if MAD-X
        # crashes, it is easier to see, where it happened:
        self._command_log(text)
        try:
            self._libmadx.input(text)
        except _rpc.RemoteProcessCrashed:
            # catch + reraise in order to shorten stack trace (~3-5 levels):
            raise RuntimeError("MAD-X has stopped working!")

    def help(self, cmd=None):
        """
        Show help about a command or list all MAD-X commands.

        :param str cmd: command name
        """
        # The case 'cmd == None' will be handled by mad_command
        # appropriately.
        self.command.help(cmd=cmd)

    def chdir(self, path):
        """
        Change the directory. Can be used as context manager.

        :param str path: new path name
        :returns: a context manager that can change the directory back
        :rtype: ChangeDirectory
        """
        # Note, that the libmadx module includes the functions 'getcwd' and
        # 'chdir' so it can be used as a valid 'os' module for the purposes
        # of ChangeDirectory:
        return ChangeDirectory(path, self._libmadx)

    def call(self, filename, chdir=False):
        """
        CALL a file in the MAD-X interpreter.

        :param str filename: file name with path
        :param bool chdir: temporarily change directory in MAD-X process
        """
        if chdir:
            dirname, basename = os.path.split(filename)
            with self.chdir(dirname):
                self.command.call(file=basename)
        else:
            self.command.call(file=filename)

    def select(self, flag, columns, pattern=[]):
        """
        Run SELECT command.

        :param str flag: one of: twiss, makethin, error, seqedit
        :param list columns: column names
        :param list pattern: selected patterns
        """
        select = self.command.select
        select(flag=flag, clear=True)
        if columns:
            select(flag=flag, column=columns)
        for p in pattern:
            select(flag=flag, pattern=p)

    def twiss(self,
              sequence=None,
              range=None,
              # *,
              # These should be passed as keyword-only parameters:
              twiss_init={},
              columns=None,
              pattern=['full'],
              **kwargs):
        """
        Run SELECT+USE+TWISS.

        :param str sequence: name of sequence
        :param list pattern: pattern to include in table
        :param list columns: columns to include in table, (may be a str)
        :param dict twiss_init: dictionary of twiss initialization variables
        :param bool chrom: Also calculate chromatic functions (slower)
        :param kwargs: further keyword arguments for the MAD-X command

        Note, that the kwargs overwrite any arguments in twiss_init.
        """
        self.select('twiss', columns=columns, pattern=pattern)
        sequence = self._use(sequence)
        twiss_init = dict((k, v) for k,v in twiss_init.items()
                          if k not in ['name','closed-orbit'])
        # explicitly specified keyword arguments overwrite values in
        # twiss_init:
        twiss_init.update(kwargs)
        self.command.twiss(sequence=sequence,
                           range=range,
                           **twiss_init)
        return self.get_table('twiss', columns)

    def survey(self,
               sequence=None,
               columns=None,
               pattern=['full'],
               **kwargs):
        """
        Run SELECT+USE+SURVEY.

        :param str sequence: name of sequence
        :param list pattern: pattern to include in table
        :param list columns: Columns to include in table
        :param kwargs: further keyword arguments for the MAD-X command
        """
        self.select('survey', pattern=pattern, columns=columns)
        self._use(sequence)
        self.command.survey(**kwargs)
        return self.get_table('survey', columns)

    def use(self, sequence):
        """
        Run USE to expand a sequence.

        :param str sequence: sequence name
        :returns: name of active sequence
        """
        self.command.use(sequence=sequence)

    def _use(self, sequence):
        """
        USE sequence if it is not active.

        :param str sequence: sequence name, may be None
        :returns: new active sequence name
        :rtype: str
        :raises RuntimeError: if there is no active sequence
        """
        try:
            active_sequence = self.active_sequence
        except RuntimeError:
            if not sequence:
                raise
            active_sequence = None
        else:
            if not sequence:
                sequence = active_sequence
        if (sequence != active_sequence
                or not self._libmadx.is_expanded(sequence)):
            self.use(sequence)
        return sequence

    def match(self,
              sequence=None,
              constraints=[],
              vary=[],
              weight=None,
              method=('lmdif', {}),
              knobfile=None,
              twiss_init={},
              **kwargs):
        """
        Perform a simple MATCH operation.

        For more advanced cases, you should issue the commands manually.

        :param str sequence: name of sequence
        :param list constraints: constraints to pose during matching
        :param list vary: knob names to be varied
        :param dict weight: weights for matching parameters
        :returns: final knob values
        :rtype: dict

        Example:

        >>> mad.match(
        ...     constraints=[
        ...         dict(range='marker1->betx',
        ...              betx=Constraint(min=1, max=3),
        ...              bety=2)
        ...     ],
        ...     vary=['qp1->k1',
        ...           'qp2->k1'])
        """
        sequence = self._use(sequence)
        twiss_init = dict((k, v) for k,v in twiss_init.items()
                          if k not in ['name','closed-orbit'])
        # explicitly specified keyword arguments overwrite values in
        # twiss_init:
        twiss_init.update(kwargs)

        command = self.command
        # MATCH (=start)
        command.match(sequence=sequence, **twiss_init)
        for c in constraints:
            command.constraint(**c)
        for v in vary:
            command.vary(name=v)
        if weight:
            command.weight(**weight)
        command(method[0], **method[1])
        command.endmatch(knobfile=knobfile)
        return dict((knob, self.evaluate(knob)) for knob in vary)

    def verbose(self, switch=True):
        """Turn verbose output on/off."""
        self.command.option(echo=switch, warn=switch, info=switch)

    def get_table(self, table, columns=None):
        """
        Get the specified table from MAD-X.

        :param str table: table name
        :returns: a proxy for the table data
        :rtype: TableProxy
        """
        proxy = TableProxy(table, self._libmadx)
        if columns is None:
            return proxy
        else:
            return proxy.copy(columns)

    @property
    def active_sequence(self):
        """Get/set the name of the active sequence."""
        return self._libmadx.get_active_sequence()

    @active_sequence.setter
    def active_sequence(self, name):
        try:
            active_sequence = self.active_sequence
        except RuntimeError:
            self.use(name)
        else:
            if active_sequence != name:
                self.use(name)

    def get_active_sequence(self):
        """
        Get a handle to the active sequence.

        :returns: a proxy object for the sequence
        :rtype: Sequence
        :raises RuntimeError: if there is no active sequence
        """
        return Sequence(self.active_sequence, self._libmadx, _check=False)

    def get_sequence(self, name):
        """
        Get a handle to the specified sequence.

        :param str name: sequence name
        :returns: a proxy object for the sequence
        :rtype: Sequence
        :raises ValueError: if a sequence name is invalid
        """
        return Sequence(name, self._libmadx)

    def has_sequence(self, sequence):
        """
        Check if model has the sequence.

        :param string sequence: sequence name to be checked.
        """
        return sequence in self.get_sequence_names()

    def get_sequences(self):
        """
        Return list of all sequences currently in memory.

        :returns: list of sequence proxy objects
        :rtype: list(Sequence)
        """
        return [Sequence(name, self._libmadx, _check=False)
                for name in self.get_sequence_names()]

    def get_sequence_names(self):
        """
        Return list of all sequences currently in memory.

        :returns: list of all sequences names
        :rtype: list(str)
        """
        return self._libmadx.get_sequences()

    def evaluate(self, cmd):
        """
        Evaluates an expression and returns the result as double.

        :param string cmd: expression to evaluate.
        :returns: numeric value of the expression
        :rtype: float
        """
        return self._libmadx.evaluate(cmd)


class Sequence(object):

    """
    MAD-X sequence representation.
    """

    def __init__(self, name, libmadx, _check=True):
        """Store sequence name."""
        self._name = name
        self._libmadx = libmadx
        if _check and not libmadx.sequence_exists(name):
            raise ValueError("Invalid sequence: {!r}".format(name))

    def __str__(self):
        """String representation."""
        return "{}({})".format(self.__class__.__name__, self._name)

    __repr__ = __str__

    @property
    def name(self):
        """Get the name of the sequence."""
        return self._name

    @property
    def beam(self):
        """Get the beam dictionary associated to the sequence."""
        return self._libmadx.get_beam(self._name)

    @property
    def twiss(self):
        """Get the TWISS results from the last calculation."""
        return TableProxy(self.twissname, self._libmadx)

    @property
    def twissname(self):
        """Get the name of the table with the TWISS results."""
        return self._libmadx.get_twiss(self._name)

    @property
    def elements(self):
        """Get list of elements."""
        return ElementList(self._name, self._libmadx, expanded=False)

    @property
    def expanded_elements(self):
        """Get list of elements in expanded sequence."""
        return ElementList(self._name, self._libmadx, expanded=True)

    def get_element_list(self):
        """
        Get list of all elements in the original sequence.

        :returns: list of elements in the original (unexpanded) sequence
        :rtype: list(dict)
        """
        return self._libmadx.get_element_list(self._name)

    def get_expanded_element_list(self):
        """
        Get list of all elements in the expanded sequence.

        :returns: list of elements in the expanded (unexpanded) sequence
        :rtype: list(dict)

        NOTE: this may very well return an empty list, if the sequence has
        not been expanded (used) yet.
        """
        return self._libmadx.get_expanded_element_list(self._name)


class ElementList(collections.Sequence):

    """
    Immutable list of beam line elements.

    Each element is a dictionary containing its properties.
    """

    def __init__(self, sequence_name, libmadx, expanded):
        """
        Initialize instance.
        """
        self._sequence_name = sequence_name
        if expanded:
            self._get_element = libmadx.get_expanded_element
            self._get_element_count = libmadx.get_expanded_element_count
            self._get_element_index = libmadx.get_expanded_element_index
            self._get_element_at = libmadx.get_expanded_element_index_by_position
        else:
            self._get_element = libmadx.get_element
            self._get_element_count = libmadx.get_element_count
            self._get_element_index = libmadx.get_element_index
            self._get_element_at = libmadx.get_element_index_by_position

    def __contains__(self, element):
        """
        Check if sequence contains element with specified name.

        Can be invoked with either the element dict or the element name.
        """
        try:
            self.index(element)
            return True
        except ValueError:
            return False

    def __getitem__(self, index):
        """Return element with specified index."""
        return self._get_element(self._sequence_name, index)

    def __len__(self):
        """Get number of elements."""
        return self._get_element_count(self._sequence_name)

    def index(self, element):
        """
        Find index of element with specified name.

        Can be invoked with either the element dict or the element name.

        :raises ValueError:
        """
        if isinstance(element, dict):
            name = element['name']
        else:
            name = element
        return self._get_element_index(self._sequence_name, name)

    def at(self, pos):
        """Find the element at specified S position."""
        return self._get_element_at(self._sequence_name, pos)


class Dict(dict):
    pass


class TableProxy(collections.Mapping):

    """
    Proxy object for lazy-loading table column data.
    """

    def __init__(self, name, libmadx, _check=True):
        """Just store the table name for now."""
        self._name = name
        self._libmadx = libmadx
        if _check and not libmadx.table_exists(name):
            raise ValueError("Invalid table: {!r}".format(name))

    def __getitem__(self, column):
        """Get the column data."""
        try:
            return self._libmadx.get_table_column(self._name, column.lower())
        except ValueError:
            raise KeyError(column)

    def __iter__(self):
        """Iterate over all column names."""
        return iter(self._libmadx.get_table_columns(self._name))

    def __len__(self):
        """Return number of columns."""
        return len(self._libmadx.get_table_columns(self._name))

    @property
    def summary(self):
        """Get the table summary."""
        return self._libmadx.get_table_summary(self._name)

    def copy(self, columns=None):
        """
        Return a frozen table with the desired columns.

        :param list columns: column names or ``None`` for all columns.
        :returns: column data
        :rtype: dict
        :raises ValueError: if the table name is invalid
        """
        if columns is None:
            columns = self
        table = Dict((column, self[column]) for column in columns)
        table.summary = self.summary
        return table
