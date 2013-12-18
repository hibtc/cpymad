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

#cython: embedsignature=True

'''
.. module:: madx
.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen.at.cern.ch>

Main module to interface with Mad-X library.

'''

from __future__ import print_function

from cern.libmadx.madx_structures cimport sequence_list, name_list, column_info, expression, char_p_array, char_array
from cern.libmadx import table

cdef extern from "madX/mad_api.h":
    sequence_list *madextern_get_sequence_list()
cdef extern from "madX/mad_core.h":
    void madx_start()
    void madx_finish()

cdef extern from "madX/mad_str.h":
    void stolower_nq(char*)
    int mysplit(char*, char_p_array*)
cdef extern from "madX/mad_eval.h":
    void pro_input(char*)

cdef extern from "madX/mad_expr.h":
    expression* make_expression(int, char**)
    double expression_value(expression*, int)
    expression* delete_expression(expression*)
cdef extern from "madX/madx.h":
    char_p_array* tmp_p_array    # temporary buffer for splits
    char_array* c_dum

cdef extern from "madX/mad_parse.h":
    void pre_split(char*, char_array*, int)


cdef madx_input(char* cmd):
    stolower_nq(cmd)
    pro_input(cmd)

import os,sys
import collections
import cern.pymad.globals
from cern.libmadx import _madx_tools

_madstarted=False

# I think this is deprecated..
_loaded_models=[]

# private utility functions
def _tmp_filename(operation):
    """
    Create a name for a temporary file.
    """
    tmpfile = operation + '.temp.tfs'
    i = 0
    while os.path.isfile(tmpfile):
        tmpfile = operation + '.' + str(i) + '.temp.tfs'
        i += 1
    return tmpfile

# main interface
class madx:
    '''
    Python class which interfaces to Mad-X library
    '''
    def __init__(self,histfile='',recursive_history=False):
        '''
        Initializing Mad-X instance

        :param str histfile: (optional) name of file which will contain all Mad-X commands.
        :param bool recursive_history: If true, history file will contain no calls to other files.
                                       Instead, recursively writing commands from these files when called.

        '''
        global _madstarted
        if not _madstarted:
            madx_start()
            _madstarted=True
        if histfile:
            self._hist=True
            self._hfile=open(histfile,'w')
            self._rechist=recursive_history
        elif cern.pymad.globals.MAD_HISTORY_BASE:
            base=cern.pymad.globals.MAD_HISTORY_BASE
            self._hist=True
            i=0
            while os.path.isfile(base+str(i)+'.madx'):
                i+=1
            self._hfile=open(base+str(i)+'.madx','w')
            self._rechist=recursive_history
        else:
            self._hist=False
            if recursive_history:
                print("WARNING: you cannot get recursive history without history file...")
            self._rechist=False

    def __del__(self):
        '''
         Closes history file
         madx_finish should
         not be called since
         other objects might still be
         running..
        '''
        if self._rechist:
            self._hfile.close()
        #madx_finish()

    def command(self,cmd):
        '''
         Send a general Mad-X command.
         Some sanity checks are performed.

         :param string cmd: command
        '''
        cmd=_madx_tools._fixcmd(cmd)
        if type(cmd)==int: # means we should not execute command
            return cmd
        if type(cmd)==list:
            for c in cmd:
                self._single_cmd(c)
        else:
            self._single_cmd(cmd)

    def _single_cmd(self,cmd):
        if self._hist:
            if cmd[-1]=='\n':
                self._writeHist(cmd)
            else:
                self._writeHist(cmd+'\n')
        if _madx_tools._checkCommand(cmd.lower()):
            cmd = cmd.encode('utf-8')
            madx_input(cmd)
        return 0

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
              sequence,
              pattern=['full'],
              columns='name,s,betx,bety,x,y,dx,dy,px,py,mux,muy,l,k1l,angle,k2l',
              madrange='',
              fname='',
              retdict=False,
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
            :param bool retdict: if true, returns tables as dictionary types
            :param dict twiss_init: dictionary of twiss initialization variables
            :param bool use: Call use before aperture.
            :param bool chrom: Also calculate chromatic functions (slower)
        '''
        self.select('twiss',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use:
            self.use(sequence)
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
        return table.get_dict_from_mem('twiss',columns,retdict)

    def survey(self,
              sequence,
              pattern=['full'],
              columns='name,l,s,angle,x,y,z,theta',
              madrange='',
              fname='',
              retdict=False,
              use=True
              ):
        '''
            Runs select+use+survey on the sequence selected

            :param string sequence: name of sequence
            :param string fname: name of file to store tfs table
            :param list pattern: pattern to include in table
            :param string/list columns: Columns to include in table
            :param bool retdict: if true, returns tables as dictionary types
            :param bool use: Call use before survey.
        '''
        tmpfile = fname or _tmp_filename('survey')
        self.select('survey',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use:
            self.use(sequence)
        self.command('survey,'+_madx_tools._add_range(madrange)+' file="'+tmpfile+'";')
        tab,param=_madx_tools._get_dict(tmpfile,retdict)
        if not fname:
            os.remove(tmpfile)
        return tab,param

    def aperture(self,
              sequence,
              pattern=['full'],
              madrange='',
              columns='name,l,angle,x,y,z,theta',
              offsets='',
              fname='',
              retdict=False,
              use=False
              ):
        '''
         Runs select+use+aperture on the sequence selected

         :param string sequence: name of sequence
         :param string fname: name of file to store tfs table
         :param list pattern: pattern to include in table
         :param list columns: columns to include in table (can also be string)
         :param bool retdict: if true, returns tables as dictionary types
         :param bool use: Call use before aperture.
        '''
        tmpfile = fname or _tmp_filename('aperture')
        self.select('aperture',pattern=pattern,columns=columns)
        self.command('set, format="12.6F";')
        if use:
            print("Warning, use before aperture is known to cause problems")
            self.use(sequence) # this seems to cause a bug?
        _cmd='aperture,'+_madx_tools._add_range(madrange)+_madx_tools._add_offsets(offsets)
        if fname:
            _cmd+=',file="'+fname+'"'
        self.command(_cmd)
        return table.get_dict_from_mem('aperture',columns,retdict)

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
                sequence=sequence,
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
        return result,initial

    # turn on/off verbose outupt..
    def verbose(self,switch):
        if switch:
            self.command("option, echo, warn, info")
        else:
            self.command("option, -echo, -warn, -info")

    def _writeHist(self,command):
        # this still brakes for "multiline commands"...
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

    def get_sequences(self):
        '''
         Returns the sequences currently in memory
        '''
        cdef sequence_list *seqs
        seqs= madextern_get_sequence_list()
        ret={}
        for i in xrange(seqs.curr):
            name = seqs.sequs[i].name.decode('utf-8')
            ret[name]={'name':name}
            if seqs.sequs[i].tw_table.name is not NULL:
                tabname = seqs.sequs[i].tw_table.name.decode('utf-8')
                ret[name]['twissname'] = tabname
                print("Table name:", tabname)
                print("Number of columns:",seqs.sequs[i].tw_table.num_cols)
                print("Number of columns (orig):",seqs.sequs[i].tw_table.org_cols)
                print("Number of rows:",seqs.sequs[i].tw_table.curr)
        return ret

    def evaluate(self, cmd):
        """
        Evaluates an expression and returns the result as double.

        :param string cmd: expression to evaluate.

        NOTE: Call this function only from within a process scope where you
        have called ``madx_start()`` first. This limitation is due to the
        use of global variables within MAD-X.

        This function uses global variables as temporaries - which is in
        general an extremely bad design choice. In this case, however, using
        local variables would only obscure the fact that MAD-X uses global
        variables internally anyway.

        """
        # TODO: not sure about the flags (the magic constants 0, 2)
        cmd = cmd.lower().encode("utf-8")
        pre_split(cmd, c_dum, 0)
        mysplit(c_dum.c, tmp_p_array)
        expr = make_expression(tmp_p_array.curr, tmp_p_array.p)
        value = expression_value(expr, 2)
        delete_expression(expr)
        return value

