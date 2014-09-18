"""
Main module to interface with Mad-X library.

The class Madx uses a subprocess to execute MAD-X library calls remotely via
a simple RPC protocol.

The remote backend is needed due to the fact that cpymad.libmadx is a low
level binding to the MAD-X library which in turn uses global variables.
This means that the cpymad.libmadx module has to be loaded within remote
processes in order to deal with several isolated instances of MAD-X in
parallel.

Furthermore, this can be used as a security enhancement: if dealing with
unverified input, we can't be sure that a faulty MAD-X function
implementation will give access to a secure resource. This can be executing
all library calls within a subprocess that does not inherit any handles.

More importantly: the MAD-X program crashes on the tinyest error. Boxing it
in a subprocess will prevent the main process from crashing as well.
"""

from __future__ import absolute_import

from functools import partial
import logging
import os
import sys
import collections

from . import _libmadx_rpc

from cern.cpymad import _madx_tools

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
        self.__dispatch(_madx_tools.mad_command(*args, **kwargs))

    def __getattr__(self, name):
        """Return a dispatcher for a specific command."""
        return partial(self.__call__, name)


# main interface
class Madx(object):
    '''
    Python class which interfaces to Mad-X library
    '''
    _hfile = None

    def __init__(self, histfile=None, libmadx=None, logger=None):
        '''
        Initializing Mad-X instance

        :param str histfile: (optional) name of file which will contain all Mad-X commands.
        :param object libmadx: :mod:`libmadx` compatible object

        '''
        self._libmadx = libmadx or _libmadx_rpc.LibMadxClient.spawn_subprocess()[0].libmadx
        if not self._libmadx.started():
            self._libmadx.start()
        self._log = logger or logging.getLogger(__name__)

        if histfile:
            self._hfile = open(histfile,'w')

    @property
    def version(self):
        """
        Get the MAD-X version.
        """
        return Version(self._libmadx.madx_release,
                       self._libmadx.madx_date)

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
        self._writeHist(text)
        self._libmadx.input(text)

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
        CALL a file in the MAD-X interpretor.

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
        select(flag=flag, column=columns)
        for p in pattern:
            select(flag=flag, pattern=p)

    default_twiss_columns = ['name', 's',
                             'betx', 'bety',
                             'x', 'y',
                             'dx', 'dy',
                             'px', 'py',
                             'mux', 'muy'
                              'l','k1l', 'angle', 'k2l']

    def twiss(self,
              sequence=None,
              pattern=['full'],
              columns=default_twiss_columns,
              range=None,
              twiss_init={},
              use=True,
              **kwargs):
        """
        Run SELECT+USE+TWISS.

        :param str sequence: name of sequence
        :param list pattern: pattern to include in table
        :param list columns: columns to include in table, (may be a str)
        :param dict twiss_init: dictionary of twiss initialization variables
        :param bool use: Call use before aperture.
        :param bool chrom: Also calculate chromatic functions (slower)
        :param kwargs: further keyword arguments for the MAD-X command

        Note, that the kwargs overwrite any arguments in twiss_init.
        """
        self.select('twiss', columns=columns, pattern=pattern)
        self.command.set(format="12.6F")
        if use and sequence:
            self.use(sequence)
        elif not sequence:
            sequence = self.active_sequence
        twiss_init = dict((k, v) for k,v in twiss_init.items()
                          if k not in ['name','closed-orbit'])
        # explicitly specified keyword arguments overwrite values in
        # twiss_init:
        twiss_init.update(kwargs)
        self.command.twiss(sequence=sequence,
                           range=range,
                           **twiss_init)
        return self.get_table('twiss')

    default_survey_columns = ['name', 'l', 's', 'angle',
                              'x', 'y', 'z', 'theta']

    def survey(self,
               sequence=None,
               pattern=['full'],
               columns=default_survey_columns,
               range=None,
               use=True,
               **kwargs):
        """
        Run SELECT+USE+SURVEY.

        :param str sequence: name of sequence
        :param list pattern: pattern to include in table
        :param list columns: Columns to include in table
        :param bool use: Call use before survey.
        :param kwargs: further keyword arguments for the MAD-X command
        """
        self.select('survey', pattern=pattern, columns=columns)
        self.command.set(format="12.6F")
        if use and sequence:
            self.use(sequence)
        self.command.survey(range=range, **kwargs)
        return self.get_table('survey')

    default_aperture_columns = ['name', 'l', 'angle'
                                'x', 'y', 'z', 'theta']

    def aperture(self,
                 sequence=None,
                 pattern=['full'],
                 range=None,
                 columns=default_aperture_columns,
                 offsets=None,
                 use=False,
                 **kwargs):
        """
        Run SELECT+USE+APERTURE.

        :param str sequence: name of sequence
        :param list pattern: pattern to include in table
        :param list columns: columns to include in table (may be a str)
        :param bool use: Call use before aperture.
        :param kwargs: further keyword arguments for the MAD-X command
        """
        self.select('aperture', pattern=pattern, columns=columns)
        self.command.set(format="12.6F")
        if use and sequence:
            self._log.warn("USE before APERTURE is known to cause problems.")
            self.use(sequence) # this seems to cause a bug?
        self.command.aperture(range=range, offsetelem=offsets, **kwargs)
        return self.get_table('aperture')

    def use(self, sequence):
        self.command.use(sequence=sequence)

    def match(self,
              sequence,
              constraints,
              vary,
              weight=None,
              method=('lmdif', {}),
              knobfile=None,
              twiss_init={},
              **kwargs):
        """
        Perform simple MATCH operation.

        :param string sequence: name of sequence
        :param list constraints: constraints to pose during matching
        :param list vary: vary commands
        :param dict weight: weights for matching parameters
        """
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

    # turn on/off verbose outupt..
    def verbose(self, switch):
        self.command.option(echo=switch, warn=switch, info=switch)

    def _writeHist(self,command):
        # this still brakes for "multiline commands"...
        if self._hfile:
            self._hfile.write(command + '\n')
            self._hfile.flush()

    def get_table(self, table):
        """
        Get the specified table columns as numpy arrays.

        :param str table: table name
        :param columns: column names
        :type columns: list or str (comma separated)

        """
        return TableProxy(table, self._libmadx)

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

    def get_elements(self):
        """
        Get list of all elements in the original sequence.

        :returns: list of elements in the original (unexpanded) sequence
        :rtype: list(dict)
        """
        return self._libmadx.get_elements(self._name)

    def get_expanded_elements(self):
        """
        Get list of all elements in the expanded sequence.

        :returns: list of elements in the expanded (unexpanded) sequence
        :rtype: list(dict)

        NOTE: this may very well return an empty list, if the sequence has
        not been expanded (used) yet.
        """
        return self._libmadx.get_expanded_elements(self._name)


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
        return dict((column, self[column]) for column in columns)
