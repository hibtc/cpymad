#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
'''
.. module:: madx
.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen.at.cern.ch>

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
'''

from __future__ import absolute_import
from __future__ import print_function

from functools import partial
import os
import sys
import collections

from . import _libmadx_rpc
from .types import Element

from cern.cpymad import _madx_tools
from cern.cpymad.types import TfsTable, TfsSummary

try:
    basestring
except NameError:
    basestring = str


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
    _hist = False
    _hfile = None
    _rechist = False

    def __init__(self, histfile=None, recursive_history=False, libmadx=None):
        '''
        Initializing Mad-X instance

        :param str histfile: (optional) name of file which will contain all Mad-X commands.
        :param bool recursive_history: If true, history file will contain no calls to other files.
                                       Instead, recursively writing commands from these files when called.
        :param object libmadx: :mod:`libmadx` compatible object

        '''
        self._libmadx = libmadx or _libmadx_rpc.LibMadxClient.spawn_subprocess()[0].libmadx
        if not self._libmadx.started():
            self._libmadx.start()

        if histfile:
            self._hist = True
            self._hfile = open(histfile,'w')
            self._rechist = recursive_history

    def __del__(self):
        """Close history file."""
        if self._hfile:
            self._hfile.close()

    @property
    def command(self):
        """
        Perform a single MAD-X command.

        :param str cmd: command name
        :param **kwargs: command parameters
        """
        return MadxCommands(self._check_command)

    def _check_command(self, cmd):
        """Execute command after performing some sanity checks."""
        if cmd.lower() in ('stop;', 'exit;'):
            print("WARNING: found quit in command: "+cmd+"\n")
            print("Please use madx.finish() or just exit python (CTRL+D)")
            print("Command ignored")
            return
        if cmd.lower().startswith('plot'):
            print("WARNING: Plot functionality does not work through pymadx")
            print("Command ignored")
            return
        self.input(cmd)

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
        fname=filename
        if not os.path.isfile(fname):
            fname=filename+'.madx'
        if not os.path.isfile(fname):
            fname=filename+'.mad'
        if not os.path.isfile(fname):
            print("ERROR: "+filename+" not found")
            return 1
        if chdir:
            dirname, basename = os.path.split(fname)
            with self.chdir(dirname):
                self.command.call(file=basename)
        else:
            self.command.call(file=fname)

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
              madrange=None,
              fname=None,
              twiss_init={},
              use=True,
              **kwargs):
        """
        Run SELECT+USE+TWISS.

        :param str sequence: name of sequence
        :param str fname: name of file to store tfs table
        :param list pattern: pattern to include in table
        :param list columns: columns to include in table, (may be a str)
        :param dict twiss_init: dictionary of twiss initialization variables
        :param bool use: Call use before aperture.
        :param bool chrom: Also calculate chromatic functions (slower)
        :param **kwargs: further keyword arguments to TWISS (betx, bety, ..).

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
                           range=madrange,
                           file=fname,
                           **twiss_init)
        return self.get_table('twiss')

    default_survey_columns = ['name', 'l', 's', 'angle',
                              'x', 'y', 'z', 'theta']

    def survey(self,
               sequence=None,
               pattern=['full'],
               columns=default_survey_columns,
               madrange=None,
               fname=None,
               use=True):
        """
        Run SELECT+USE+SURVEY.

        :param str sequence: name of sequence
        :param str fname: name of file to store tfs table
        :param list pattern: pattern to include in table
        :param list columns: Columns to include in table
        :param bool use: Call use before survey.
        """
        self.select('survey', pattern=pattern, columns=columns)
        self.command.set(format="12.6F")
        if use and sequence:
            self.use(sequence)
        self.command.survey(range=madrange, file=fname)
        return self.get_table('survey')

    default_aperture_columns = ['name', 'l', 'angle'
                                'x', 'y', 'z', 'theta']

    def aperture(self,
                 sequence=None,
                 pattern=['full'],
                 madrange='',
                 columns=default_aperture_columns,
                 offsets=None,
                 fname=None,
                 use=False):
        """
        Run SELECT+USE+APERTURE.

        :param str sequence: name of sequence
        :param str fname: name of file to store tfs table
        :param list pattern: pattern to include in table
        :param list columns: columns to include in table (may be a str)
        :param bool use: Call use before aperture.
        """
        self.select('aperture', pattern=pattern, columns=columns)
        self.command.set(format="12.6F")
        if use and sequence:
            print("Warning, use before aperture is known to cause problems")
            self.use(sequence) # this seems to cause a bug?
        self.command.aperture(range=madrange, offsetelem=offsets, file=fname)
        return self.get_table('aperture')

    def use(self, sequence):
        self.command.use(sequence=sequence)

    def match(self,
              sequence,
              constraints,
              vary,
              weight=None,
              method=('lmdiff', {}),
              fname=None,
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
        command.endmatch(knobfile=fname)

    # turn on/off verbose outupt..
    def verbose(self, switch):
        self.command.option(echo=switch, warn=switch, info=switch)

    def _writeHist(self,command):
        # this still brakes for "multiline commands"...
        if not self._hfile:
            return
        if self._rechist and command.split(',')[0].strip().lower()=='call':
            cfile=command.split(',')[1].strip().strip('file=').strip('FILE=').strip(';\n').strip('"').strip("'")
            if sys.flags.debug:
                print("DBG: call file ",cfile)
            fin=open(cfile,'r')
            for l in fin:
                self._writeHist(l+'\n')
        else:
            self._hfile.write(command)
            self._hfile.flush()

    def get_table(self, table):
        """
        Get the specified table columns as numpy arrays.

        :param str table: table name
        :param columns: column names
        :type columns: list or str (comma separated)

        """
        return Table(table, self._libmadx)

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
        return Table(self.twissname, self._libmadx)

    @property
    def twissname(self):
        """Get the name of the table with the TWISS results."""
        return self._libmadx.get_twiss(self._name)

    def get_elements(self):
        """
        Get list of all elements in the original sequence.

        :returns: list of elements in the original (unexpanded) sequence
        :rtype: list(Element)
        """
        return [Element(elem)
                for elem in self._libmadx.get_elements(self._name)]

    def get_expanded_elements(self):
        """
        Get list of all elements in the expanded sequence.

        :returns: list of elements in the expanded (unexpanded) sequence
        :rtype: list(Element)

        NOTE: this may very well return an empty list, if the sequence has
        not been expanded (used) yet.
        """
        return [Element(elem)
                for elem in self._libmadx.get_expanded_elements(self._name)]


class Table(object):

    """
    MAD-X table access class.
    """

    def __init__(self, name, libmadx, _check=True):
        """Just store the table name for now."""
        self._name = name
        self._libmadx = libmadx
        if _check and not libmadx.table_exists(name):
            raise ValueError("Invalid table: {!r}".format(name))

    def __iter__(self):
        """Old style access."""
        columns = self.columns
        try:
            summary = self.summary
        except ValueError:
            summary = None
        return iter((columns, summary))

    @property
    def name(self):
        """Get the table name."""
        return self._name

    @property
    def columns(self):
        """Get a lazy accessor for the table columns."""
        return TableColumns(self.name, self._libmadx)

    @property
    def summary(self):
        """Get the table summary."""
        return TfsSummary(self._libmadx.get_table_summary(self.name))


class TableColumns(object):

    """
    Lazy accessor for table column data.
    """

    def __init__(self, table, libmadx):
        """Store tabe name and libmadx connection."""
        self._table = table
        self._libmadx = libmadx

    def __getattr__(self, column):
        """Get the column data."""
        try:
            return self._libmadx.get_table_column(self._table, column.lower())
        except ValueError:
            raise AttributeError(column)

    def __getitem__(self, column):
        """Get the column data."""
        try:
            return self._libmadx.get_table_column(self._table, column.lower())
        except ValueError:
            raise KeyError(column)

    def __iter__(self):
        """Get a list of all column names."""
        return iter(self._libmadx.get_table_columns(self._table))

    def freeze(self, columns=None):
        """
        Return a frozen table with the desired columns.

        :param list columns: column names or ``None`` for all columns.
        :returns: column data
        :rtype: TfsTable
        :raises ValueError: if the table name is invalid
        """
        if columns is None:
            columns = self
        return TfsTable(dict((column, self[column]) for column in columns))
