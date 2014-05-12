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
'''

from __future__ import absolute_import
from __future__ import print_function

import os, sys
import collections

from . import _libmadx_rpc
from .types import Element

from cern.cpymad import _madx_tools
from cern.cpymad.types import TfsTable, TfsSummary

try:
    basestring
except NameError:
    basestring = str

# private utility functions
def _tmp_filename(operation, suffix='.temp.tfs'):
    """
    Create a name for a temporary file.
    """
    tmpfile = operation + suffix
    i = 0
    while os.path.isfile(tmpfile):
        tmpfile = operation + '.' + str(i) + suffix
        i += 1
    return tmpfile


def _check_command(cmd):
    ''' give the lowercase version of the command
    this function does some sanity checks...'''
    if cmd in ('stop;', 'exit;'):
        print("WARNING: found quit in command: "+cmd+"\n")
        print("Please use madx.finish() or just exit python (CTRL+D)")
        print("Command ignored")
        return False
    if cmd.startswith('plot'):
        print("WARNING: Plot functionality does not work through pymadx")
        print("Command ignored")
        return False
    # All checks passed..
    return True


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
        self._libmadx = libmadx or _libmadx_rpc.LibMadxClient.spawn_subprocess().libmadx
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

    def command(self, cmd):
        '''
        Perform sequence of MAD-X commands.

        :param string cmd: command sequence

        This function can take multiple commands separated by a semi-colon.

        '''
        for c in cmd.split(';'):
            c = c.strip() + ';'
            if _check_command(c.lower()):
                self._writeHist(c + '\n')
                self._libmadx.input(c)

    def help(self,cmd=''):
        if cmd:
            print("Information about command: "+cmd.strip())
            cmd='help,'+cmd
        else:
            cmd='help'
            print("Available commands in Mad-X: ")
        self.command(cmd)

    def call(self,filename):
        '''
         Call a file

         :param string filename: Name of input file to call
        '''
        fname=filename
        if not os.path.isfile(fname):
            fname=filename+'.madx'
        if not os.path.isfile(fname):
            fname=filename+'.mad'
        if not os.path.isfile(fname):
            print("ERROR: "+filename+" not found")
            return 1
        cmd='call,file="'+fname+'"'
        self.command(cmd)
    ##
    # @brief run select command for a flag..
    # @param flag [string] the flag to run select on
    # @param pattern [list] the new pattern
    # @param columns [list/string] the columns you want for the flag
    def select(self,flag,columns,pattern=[]):
        self.command('SELECT, FLAG='+flag+', CLEAR;')
        if type(columns)==list:
            clms=', '.join(columns)
        else:
            clms=columns
        self.command('SELECT, FLAG='+flag+', COLUMN='+clms+';')
        for p in pattern:
            self.command('SELECT, FLAG='+flag+', PATTERN='+p+';')

    def twiss(self,
              sequence=None,
              pattern=['full'],
              columns='name,s,betx,bety,x,y,dx,dy,px,py,mux,muy,l,k1l,angle,k2l',
              madrange='',
              fname='',
              betx=None,
              bety=None,
              alfx=None,
              alfy=None,
              twiss_init=None,
              chrom=True,
              use=True
              ):
        '''

            Runs select+use+twiss on the sequence selected

            :param string sequence: name of sequence
            :param string fname: name of file to store tfs table
            :param list pattern: pattern to include in table
            :param string columns: columns to include in table, can also be a list of strings
            :param dict twiss_init: dictionary of twiss initialization variables
            :param bool use: Call use before aperture.
            :param bool chrom: Also calculate chromatic functions (slower)
        '''
        self.select('twiss',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use and sequence:
            self.use(sequence)
        elif not sequence:
            sequence = self.active_sequence
        _tmpcmd='twiss, sequence='+sequence+','+_madx_tools._add_range(madrange)
        if chrom:
            _tmpcmd+=',chrom'
        if _tmpcmd[-1]==',':
            _tmpcmd=_tmpcmd[:-1]
        if fname: # we only need this if user wants the file to be written..
            _tmpcmd+=', file="'+fname+'"'
        for i_var,i_val in {'betx':betx,'bety':bety,'alfx':alfx,'alfy':alfy}.items():
            if i_val is not None:
                _tmpcmd+=','+i_var+'='+str(i_val)
        if twiss_init:
            for i_var,i_val in twiss_init.items():
                if i_var not in ['name','closed-orbit']:
                    if i_val is True:
                        _tmpcmd+=','+i_var
                    else:
                        _tmpcmd+=','+i_var+'='+str(i_val)
        self.command(_tmpcmd+';')
        return self.get_table('twiss')

    def survey(self,
              sequence=None,
              pattern=['full'],
              columns='name,l,s,angle,x,y,z,theta',
              madrange='',
              fname='',
              use=True
              ):
        '''
            Runs select+use+survey on the sequence selected

            :param string sequence: name of sequence
            :param string fname: name of file to store tfs table
            :param list pattern: pattern to include in table
            :param string/list columns: Columns to include in table
            :param bool use: Call use before survey.
        '''
        tmpfile = fname or _tmp_filename('survey')
        self.select('survey',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use and sequence:
            self.use(sequence)
        self.command('survey,'+_madx_tools._add_range(madrange)+' file="'+tmpfile+'";')
        return self.get_table('survey')

    def aperture(self,
              sequence=None,
              pattern=['full'],
              madrange='',
              columns='name,l,angle,x,y,z,theta',
              offsets='',
              fname='',
              use=False
              ):
        '''
         Runs select+use+aperture on the sequence selected

         :param string sequence: name of sequence
         :param string fname: name of file to store tfs table
         :param list pattern: pattern to include in table
         :param list columns: columns to include in table (can also be string)
         :param bool use: Call use before aperture.
        '''
        tmpfile = fname or _tmp_filename('aperture')
        self.select('aperture',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use and sequence:
            print("Warning, use before aperture is known to cause problems")
            self.use(sequence) # this seems to cause a bug?
        _cmd='aperture,'+_madx_tools._add_range(madrange)+_madx_tools._add_offsets(offsets)
        if fname:
            _cmd+=',file="'+fname+'"'
        self.command(_cmd)
        return self.get_table('aperture')

    def use(self,sequence):
        self.command('use, sequence='+sequence+';')

    @staticmethod
    def matchcommand(
        sequence,
        constraints,
        vary,
        weight=None,
        method=['lmdif'],
        fname='',
        betx=None,
        bety=None,
        alfx=None,
        alfy=None,
        twiss_init=None):
        """
        Prepare a match command sequence.

        :param string sequence: name of sequence
        :param list constraints: constraints to pose during matching
        :param list vary: vary commands (can also be dict)
        :param dict weight: weights for matching parameters
        :param list method: Which method to apply

        Each item of constraints must be a list or dict directly passable
        to _mad_command().

        If vary is a list each entry must either be list or a dict which can
        be passed to _mad_command(). Otherwise the value is taken to be the
        NAME of the variable.
        If vary is a dict the key corresponds to the NAME. Its value is a
        list or dict passable to _mad_command(). If this is not the case the
        value is taken as the STEP value.

        Examples:

        >>> print(madx.matchcommand(
        ...     'lhc',
        ...     constraints=[{'betx':3, 'range':'#e'}, [('bety','<',3)]],
        ...     vary=['K1', {'name':'K2', 'step':1e-6}],
        ...     weight=dict(betx=1, bety=2),
        ...     method=['lmdif', dict(calls=100, tolerance=1e-6)]
        ... ).rstrip())
        match, sequence=lhc;
        constraint, betx=3, range=#e;
        constraint, bety<3;
        vary, name=K1;
        vary, name=K2, step=1e-06;
        weight, betx=1, bety=2;
        lmdif, calls=100, tolerance=1e-06;
        endmatch;

        >>> print(madx.matchcommand(
        ...     'lhc',
        ...     constraints=[{'betx':3, 'range':'#e'}],
        ...     vary={'K1': {'upper':3}, 'K2':1e-6},
        ...     fname='knobs.txt'
        ... ).rstrip())
        match, sequence=lhc;
        constraint, betx=3, range=#e;
        vary, upper=3, name=K1;
        vary, name=K2, step=1e-06;
        lmdif;
        endmatch, knobfile=knobs.txt;

        """
        if not twiss_init:
            twiss_init = {}
        for k,v in {'betx':betx,'bety':bety,'alfx':alfx,'alfy':alfy}.items():
            if v is not None:
                twiss_init[k] = v

        # MATCH (=start)
        cmd = _madx_tools._mad_command('match', ('sequence', sequence), **twiss_init)

        # CONSTRAINT
        assert isinstance(constraints, collections.Sequence)
        for c in constraints:
            cmd += _madx_tools._mad_command_unpack('constraint', c)

        # VARY
        if isinstance(vary, collections.Mapping):
            for k,v in _madx_tools._sorted_items(vary):
                try:
                    cmd += _madx_tools._mad_command_unpack('vary', v, name=k)
                except TypeError:
                    cmd += _madx_tools._mad_command('vary', name=k, step=v)
        elif isinstance(vary, collections.Sequence):
            for v in vary:
                if isinstance(v, basestring):
                    cmd += _madx_tools._mad_command('vary', name=v)
                else:
                    cmd += _madx_tools._mad_command_unpack('vary', v)
        else:
            raise TypeError("vary must be list or dict.")

        # WEIGHT
        if weight:
            cmd += _madx_tools._mad_command_unpack('weight', weight)

        # METHOD
        cmd += _madx_tools._mad_command_unpack(*method)

        # ENDMATCH
        if fname:
            cmd += _madx_tools._mad_command('endmatch', knobfile=fname)
        else:
            cmd += _madx_tools._mad_command('endmatch')
        return cmd


    def match(
            self,
            sequence,
            constraints,
            vary,
            weight=None,
            method=['lmdif'],
            fname='',
            betx=None,
            bety=None,
            alfx=None,
            alfy=None,
            twiss_init=None,
            retdict=False):
        """
        Perform match operation.

        @param sequence [string] name of sequence
        @param constraints [list] constraints to pose during matching
        @param vary [list or dict] vary commands
        @param weight [dict] weights for matching parameters

        For further reference, see madx.matchcommand().

        """
        tmpfile = fname or _tmp_filename('match')

        cmd = self.matchcommand(
                sequence=sequence or self.active_sequence,
                constraints=constraints,
                vary=vary,
                weight=weight,
                method=method,
                fname=tmpfile,
                betx=betx, bety=bety,
                alfx=alfx, alfy=alfy,
                twiss_init=twiss_init)

        self.command(cmd)
        result,initial=_madx_tools._read_knobfile(tmpfile, retdict)
        if not fname:
            os.remove(tmpfile)
        return (result,initial)

    # turn on/off verbose outupt..
    def verbose(self,switch):
        if switch:
            self.command("option, echo, warn, info")
        else:
            self.command("option, -echo, -warn, -info")

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
        :param bool retdict: return plain dictionaries
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
