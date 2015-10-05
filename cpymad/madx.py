# encoding: utf-8
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

    """
    Context manager for temporarily changing current working directory in the
    context of the given ``os`` module.

    :param str path: new path name
    :param _os: module with ``getcwd`` and ``chdir`` functions
    """

    def __init__(self, path, _os):
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

    def __init__(self, file, prefix='', suffix='\n'):
        """Create CommandLog from file instance."""
        self._file = file
        self._prefix = prefix
        self._suffix = suffix

    def __call__(self, command):
        """Log a single history line and flush to file immediately."""
        self._file.write(self._prefix + command + self._suffix)
        self._file.flush()


class Madx(object):

    """
    Each instance controls one MAD-X interpretor.

    This class aims to expose a pythonic interface to the full functionality
    of the MAD-X library. For example, when you call the ``twiss`` method, you
    get numpy arrays containing the information from the table generated.
    Furthermore, we try to reduce the amount of commands needed by combining
    e.g. USE, SELECT, and TWISS into the ``twiss`` method itself, and define
    reasonable default patterns/columns.

    The following very simple example demonstrates basic usage::

        m = Madx()
        m.call('my_stuff.madx')
        twiss = m.twiss('myseq1')

        matplotlib.pyplot.plt(twiss['s'], twiss['betx'])

    Note that only few MAD-X commands are exposed as :class:`Madx` methods.
    For the rest, there is :meth:`Madx.command` which can be used to execute
    arbitrary MAD-X commands::

        m.command.beam(sequence='myseq1', particle='PROTON')

    This will work for the majority of cases. However, in some instances it
    may be necessary to forgo some of the syntactic sugar that
    :meth:`Madx.command` provides. For example, the ``global`` command (part
    of matching) can not be accessed as attribute since it is a python
    keyword. This can be handled as follows::

        m.command('global', sequence=cassps, Q1=26.58)

    Composing MAD-X commands can be a bit tricky at times — partly because of
    some inconsistencies in the MAD-X language. For those cases where
    ``command`` fails to do the right thing or where you simply need more fine
    grained control over command composition there are more powerful syntaxes
    available::

        # Multiple positional arguments are just concatenated with commas in
        # between:
        m.command('global', 'sequence=cassps', Q1=26.58)
        m.command('global, sequence=cassps',  'Q1=26.58')

        # Issue a plain text command, don't forget the semicolon!
        m.input('FOO, BAR=[baz], QUX=<NORF>;')

    By default :class:`Madx` uses a subprocess to execute MAD-X library calls
    remotely via a simple RPC protocol which is defined in :mod:`_rpc`. If
    required this behaviour can be customized by passing a custom ``libmadx``
    object to the constructor. This object must expose an interface similar to
    :mod:`libmadx`.
    """

    def __init__(self, libmadx=None, command_log=None, error_log=None,
                 **Popen_args):
        """
        Initialize instance variables.

        :param libmadx: :mod:`libmadx` compatible object
        :param command_log: Log all MAD-X commands issued via cpymad.
        :param error_log: logger instance ``logging.Logger``
        :param Popen_args: Additional parameters to ``subprocess.Popen``

        Note that ``command_log`` can be either a filename or a callable. For
        example:

            m1 = Madx(command_log=print)

            m2 = Madx(command_log=CommandLog(sys.stderr))

        Of course, in python2 the first example requires ``from __future__
        import print_function`` to be in effect.

        If ``libmadx`` is NOT specified, a new MAD-X interpretor will
        automatically be spawned. This is what you will mostly want to do. In
        this case any additional keyword arguments are forwarded to
        ``subprocess.Popen``. The most prominent use case for this is to
        redirect or suppress the MAD-X standard I/O::

            m = Madx(stdout=False)

            with open('madx_output.log', 'w') as f:
                m = Madx(stdout=f)

            m = Madx(stdout=subprocess.PIPE)
            f = m._process.stdout
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
            self._service, self._process = \
                _rpc.LibMadxClient.spawn_subprocess(**Popen_args)
            libmadx = self._service.libmadx
        if not libmadx.is_started():
            libmadx.start()
        # init instance variables:
        self._libmadx = libmadx
        self._command_log = command_log
        self._error_log = error_log

    def __bool__(self):
        """Check if MAD-X is up and running."""
        try:
            return self._libmadx.is_started()
        except (_rpc.RemoteProcessClosed, _rpc.RemoteProcessCrashed):
            return False

    __nonzero__ = __bool__      # alias for python2 compatibility

    @property
    def version(self):
        """Get the MAD-X version."""
        return Version(self._libmadx.get_version_number(),
                       self._libmadx.get_version_date())

    @property
    def command(self):
        """
        Perform a single MAD-X command.

        :param str cmd: command name
        :param kwargs: command parameters
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

    @property
    def globals(self):
        """
        Get a dict-like interface to global MAD-X variables.
        """
        return VarListProxy(self._libmadx)

    @property
    def elements(self):
        """
        Get a dict-like interface to globally visible elements.
        """
        return GlobalElementList(self._libmadx)

    def set_value(self, name, value):
        """
        Set a variable value ("=" operator in MAD-X).

        Example:

            >>> madx.set_value('R1QS1->K1', '42')
            >>> madx.evaluate('R1QS1->K1')
            42
        """
        self.input(name + ' = ' + str(value) + ';')

    def set_expression(self, name, expr):
        """
        Set a variable expression (":=" operator in MAD-X).

        Example:

            >>> madx.set_expression('FOO', 'BAR')
            >>> madx.set_value('BAR', 42)
            >>> madx.evaluate('FOO')
            42
            >>> madx.set_value('BAR', 43)
            >>> madx.evaluate('FOO')
            43
        """
        self.input(name + ' := ' + str(expr) + ';')

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

        Note that the kwargs overwrite any arguments in twiss_init.
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
                sequence = active_sequence.name
        if (sequence != active_sequence
                or not self._libmadx.is_sequence_expanded(sequence)):
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
        :param str knobfile: file to write the knob values to
        :param dict twiss_init: initial twiss parameters
        :param dict kwargs: further keyword arguments for the MAD-X command
        :returns: final knob values
        :rtype: dict

        Example:

        >>> from cpymad.madx import Madx
        >>> from cpymad.types import Constraint
        >>> m = Madx()
        >>> m.call('sequence.madx')
        >>> twiss_init = {'betx': 1, 'bety': 2, 'alfx': 3, 'alfy': 4}
        >>> m.match(
        ...     sequence='mysequence',
        ...     constraints=[
        ...         dict(range='marker1',
        ...              betx=Constraint(min=1, max=3),
        ...              bety=2)
        ...     ],
        ...     vary=['qp1->k1',
        ...           'qp2->k1'],
        ...     twiss_init=twiss_init,
        ... )
        >>> tw = m.twiss('mysequence', twiss_init=twiss_init)
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
        """The active :class:`Sequence` (may be None)."""
        try:
            return Sequence(self._libmadx.get_active_sequence_name(),
                            self._libmadx,
                            _check=False)
        except RuntimeError:
            return None

    @active_sequence.setter
    def active_sequence(self, sequence):
        if isinstance(sequence, Sequence):
            name = sequence.name
        elif isinstance(sequence, basestring):
            name = sequence
        try:
            active_sequence = self.active_sequence
        except RuntimeError:
            self.use(name)
        else:
            if active_sequence.name != name:
                self.use(name)

    def evaluate(self, cmd):
        """
        Evaluates an expression and returns the result as double.

        :param str cmd: expression to evaluate.
        :returns: numeric value of the expression
        :rtype: float
        """
        return self._libmadx.evaluate(cmd)

    @property
    def sequences(self):
        """A dict like view of all sequences in memory."""
        return SequenceMap(self._libmadx)

    @property
    def tables(self):
        """A dict like view of all tables in memory."""
        return TableMap(self._libmadx)


def _map_repr(self):
    """String representation of a custom mapping object."""
    return str(dict(self))


class SequenceMap(collections.Mapping):

    """
    A dict like view of all sequences (:class:`Sequence`) in memory.
    """

    def __init__(self, libmadx):
        self._libmadx = libmadx

    __repr__ = _map_repr
    __str__ = _map_repr

    def __iter__(self):
        return iter(self._libmadx.get_sequence_names())

    def __getitem__(self, name):
        try:
            return Sequence(name, self._libmadx)
        except ValueError:
            raise KeyError

    def __contains__(self, name):
        return self._libmadx.sequence_exists(name)

    def __len__(self):
        return self._libmadx.get_sequence_count()


class TableMap(collections.Mapping):

    """
    A dict like view of all tables (:class:`TableProxy`) in memory.
    """

    def __init__(self, libmadx):
        self._libmadx = libmadx

    __repr__ = _map_repr
    __str__ = _map_repr

    def __iter__(self):
        return iter(self._libmadx.get_table_names())

    def __getitem__(self, name):
        try:
            return TableProxy(name, self._libmadx)
        except ValueError:
            raise KeyError

    def __contains__(self, name):
        return self._libmadx.table_exists(name)

    def __len__(self):
        return self._libmadx.get_table_count()


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
        return "<{}: {}>".format(self.__class__.__name__, self._name)

    __repr__ = __str__

    @property
    def name(self):
        """Get the name of the sequence."""
        return self._name

    @property
    def beam(self):
        """Get the beam dictionary associated to the sequence."""
        return self._libmadx.get_sequence_beam(self._name)

    @property
    def twiss_table(self):
        """Get the TWISS results from the last calculation."""
        return TableProxy(self.twiss_table_name, self._libmadx)

    @property
    def twiss_table_name(self):
        """Get the name of the table with the TWISS results."""
        return self._libmadx.get_sequence_twiss_table_name(self._name)

    @property
    def elements(self):
        """Get list of elements."""
        return ElementList(self._libmadx, self._name)

    @property
    def expanded_elements(self):
        """Get list of elements in expanded sequence."""
        return ExpandedElementList(self._libmadx, self._name)


class BaseElementList(object):

    """
    Immutable list of beam line elements.

    Each element is a dictionary containing its properties.
    """

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
        if isinstance(index, (dict, basestring)):
            # allow element names to be passed for convenience:
            index = self.index(index)
        # _get_element accepts indices in the range [0, len-1]. The following
        # extends the accepted range to [-len, len+1], just like for lists:
        if index < 0:
            index += len(self)
        return self._get_element(index)

    def __len__(self):
        """Get number of elements."""
        return self._get_element_count()

    def index(self, element):
        """
        Find index of element with specified name.

        Can be invoked with either the element dict or the element name.

        :raises ValueError: if the element is not found
        """
        if isinstance(element, dict):
            name = element['name']
        else:
            name = element
        if len(self) == 0:
            raise ValueError('Empty element list.')
        if name == '#s':
            return 0
        elif name == '#e':
            return len(self) - 1
        index = self._get_element_index(name)
        if index == -1:
            raise ValueError("Element not in list: {!r}".format(name))
        return index


class ElementList(BaseElementList, collections.Sequence):

    def __init__(self, libmadx, sequence_name):
        """
        Initialize instance.
        """
        self._libmadx = libmadx
        self._sequence_name = sequence_name

    def at(self, pos):
        """Find the element at specified S position."""
        return self._get_element_at(pos)

    def _get_element(self, element_index):
        return self._libmadx.get_element(self._sequence_name, element_index)

    def _get_element_count(self):
        return self._libmadx.get_element_count(self._sequence_name)

    def _get_element_index(self, element_name):
        return self._libmadx.get_element_index(self._sequence_name, element_name)

    def _get_element_at(self, pos):
        return self._libmadx.get_element_index_by_position(self._sequence_name, pos)


class ExpandedElementList(ElementList):

    def _get_element(self, element_index):
        return self._libmadx.get_expanded_element(self._sequence_name, element_index)

    def _get_element_count(self):
        return self._libmadx.get_expanded_element_count(self._sequence_name)

    def _get_element_index(self, element_name):
        return self._libmadx.get_expanded_element_index(self._sequence_name, element_name)

    def _get_element_at(self, pos):
        return self._libmadx.get_expanded_element_index_by_position(self._sequence_name, pos)


class GlobalElementList(BaseElementList, collections.Mapping):

    """
    Provides dict-like access to MAD-X global elements.
    """

    def __init__(self, libmadx):
        self._libmadx = libmadx
        self._get_element = libmadx.get_global_element
        self._get_element_count = libmadx.get_global_element_count
        self._get_element_index = libmadx.get_global_element_index

    def __iter__(self):
        for i in range(len(self)):
            yield self._libmadx.get_global_element_name(i)


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
        return iter(self._libmadx.get_table_column_names(self._name))

    def __len__(self):
        """Return number of columns."""
        return len(self._libmadx.get_table_column_names(self._name))

    @property
    def summary(self):
        """Get the table summary."""
        return self._libmadx.get_table_summary(self._name)

    @property
    def range(self):
        """Get the element names (first, last) of the valid range."""
        row_count = self._libmadx.get_table_row_count(self._name)
        range = (0, row_count-1)
        return tuple(self._libmadx.get_table_row_names(self._name, range))

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


class VarListProxy(collections.MutableMapping):

    """
    Provides dict-like access to MAD-X global variables.
    """

    def __init__(self, libmadx):
        self._libmadx = libmadx

    def __getitem__(self, name):
        return self._libmadx.get_var(name)

    def __setitem__(self, name, value):
        self._libmadx.set_var(name, value)

    def __delitem__(self, name):
        raise NotImplementedError("Currently, can't erase a MAD-X global.")

    def __iter__(self):
        return iter(self._libmadx.get_globals())

    def __len__(self):
        return self._libmadx.num_globals()


class Metadata(object):

    """MAD-X metadata (license info, etc)."""

    __title__ = 'MAD-X'

    @property
    def __version__(self):
        return self._get_libmadx().get_version_number()

    __summary__ = (
        'MAD is a project with a long history, aiming to be at the '
        'forefront of computational physics in the field of particle '
        'accelerator design and simulation. The MAD scripting language '
        'is de facto the standard to describe particle accelerators, '
        'simulate beam dynamics and optimize beam optics.'
    )

    __support__ = 'mad@cern.ch'

    __uri__ = 'http://madx.web.cern.ch/madx/'

    __credits__ = (
        'MAD-X is developed at CERN and has many contributors. '
        'For more information see:\n'
        '\n'
        'http://madx.web.cern.ch/madx/www/contributors.html'
    )

    def get_copyright_notice(self):
        from pkg_resources import resource_string
        return resource_string('cpymad', 'COPYING/madx.rst')

    _libmadx = None

    def _get_libmadx(self):
        if not self._libmadx:
            svc, proc = _rpc.LibMadxClient.spawn_subprocess()
            self._libmadx = svc.libmadx
        return self._libmadx


metadata = Metadata()
