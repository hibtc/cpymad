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

'''

from cern.cpymad.madx_structures cimport sequence_list, name_list
cdef extern from "madextern.h":
    void madextern_start()
    void madextern_end()
    void madextern_input(char*)
    sequence_list *madextern_get_sequence_list()
    #table *getTable()

import os,sys
from pymad.io import tfs,tfsDict

_madstarted=False

_loaded_models=[]

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
            madextern_start()
            _madstarted=True
        if histfile:
            self._hist=True
            self._hfile=file(histfile,'w')
            self._rechist=recursive_history
        else:
            self._hist=False
            if recursive_history:
                print("WARNING: you cannot get recursive history without history file...")
            self._rechist=False
    
    def __del__(self):
        '''
         Closes history file
         madextern_end should
         not be called since
         other objects might still be
         running..
        '''
        if self._rechist:
            self._hfile.close()
        #madextern_end()
    
    # give lowercase version of command here..
    def _checkCommand(self,cmd):
        if "stop;" in cmd or "exit;" in cmd:
            print("WARNING: found quit in command: "+cmd+"\n")
            print("Please use madx.finish() or just exit python (CTRL+D)")
            print("Command ignored")
            return False
        if cmd.split(',')>0 and "plot" in cmd.split(',')[0]:
            print("WARNING: Plot functionality does not work through pymadx")
            print("Command ignored")
            return False
        # All checks passed..
        return True

    def command(self,cmd):
        '''
         Send a general Mad-X command. 
         Some sanity checks are performed.
         
         :param string cmd: command
        '''
        cmd=self._fixcmd(cmd)
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
        if self._checkCommand(cmd.lower()):
            madextern_input(cmd)
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
    def select(self,flag,pattern,columns):
        self.command('SELECT, FLAG='+flag+', CLEAR;')
        if type(columns)==list:
            #clms=','.join(str(c) for c in columns)
            clms=''
            for c in columns:
                clms+=c+','
            clms=clms[:-1]
        else:
            clms=columns
        self.command('SELECT, FLAG='+flag+', PATTERN='+pattern[0]+', COLUMN='+clms+';')
        for i in range(1,len(pattern)):
            self.command('SELECT, FLAG='+flag+', PATTERN='+pattern[i]+';')
            
    ##
    # @brief Runs select+use+twiss on the sequence selected
    # 
    # @param sequence [string] name of sequence
    # @param fname [string,optional] name of file to store tfs table
    # @param pattern [list, optional] pattern to include in table
    # @param columns [list or string, optional] columns to include in table
    # @param retdict [bool] if true, returns tables as dictionary types
    def twiss(self,
              sequence,
              pattern=['full'],
              columns='name,s,betx,bety,x,y,dx,dy,px,py,mux,muy',
              fname='',
              retdict=False
              ):
        if fname:
            tmpfile=fname
        else:
            tmpfile='twiss.temp.tfs'
            i=0
            while os.path.isfile(tmpfile):
                tmpfile='twiss.'+str(i)+'.temp.tfs'
                i+=1
        self.select('twiss',pattern,columns)
        self.command('set, format="12.6F";')
        self.use(sequence)
        self.command('twiss, sequence='+sequence+', file="'+tmpfile+'";')
        tab,param=_get_dict(tmpfile,retdict)
        if not fname:
            os.remove(tmpfile)
        return tab,param
    
    def survey(self,
              sequence,
              pattern=['full'],
              columns='name,l,angle,x,y,z,theta',
              fname='',
              retdict=False
              ):
        '''
         Runs select+use+survey on the sequence selected
         
         @param sequence [string] name of sequence
         @param fname [string,optional] name of file to store tfs table
         @param pattern [list, optional] pattern to include in table
         @param columns [string or list, optional] columns to include in table
        '''
        if fname:
            tmpfile=fname
        else:
            tmpfile='survey.temp.tfs'
            i=0
            while os.path.isfile(tmpfile):
                tmpfile='survey.'+str(i)+'.temp.tfs'
                i+=1
        self.select('survey',pattern,columns)
        self.command('set, format="12.6F";')
        self.use(sequence)
        self.command('survey, file="'+tmpfile+'";')
        tab,param=_get_dict(tmpfile,retdict)
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
              retdict=False
              ):
        '''
         Runs select+use+aperture on the sequence selected
         
         @param sequence [string] name of sequence
         @param fname [string,optional] name of file to store tfs table
         @param pattern [list, optional] pattern to include in table
         @param columns [string or list, optional] columns to include in table
        '''
        if fname:
            tmpfile=fname
        else:
            tmpfile='survey.temp.tfs'
            i=0
            while os.path.isfile(tmpfile):
                tmpfile='survey.'+str(i)+'.temp.tfs'
                i+=1
        self.select('aperture',pattern,columns)
        self.command('set, format="12.6F";')
        self.use(sequence)
        self.command('aperture,'+_add_range(madrange)+'file="'+tmpfile+'";')
        tab,param=_get_dict(tmpfile,retdict)
        if not fname:
            os.remove(tmpfile)
        return tab,param
        
    def use(self,sequence):
        self.command('use, sequence='+sequence+';')
    
    def list_of_models(self):
        global _loaded_models
        return _loaded_models
    
    def append_model(self,model_name):
        global _loaded_models
        if model_name in _loaded_models:
            raise ValueError("You cannot load the same module twice!")
        _loaded_models.append(model_name)
    
    # turn on/off verbose outupt..
    def verbose(self,switch):
        if switch:
            self.command("OPTION, ECHO, WARN, INFO;")
        else:
            self.command("OPTION, -ECHO, -WARN, -INFO;")

    def _fixcmd(self,cmd):
        if type(cmd)!=str and type(cmd)!=unicode:
            raise TypeError("ERROR: input must be a string, not "+str(type(cmd)))
        if len(cmd.strip())==0:
            return 0
        if cmd.strip()[-1]!=';':
            cmd+=';'
        # for very long commands (probably parsed in from a file)
        # we split and only run one line at the time.
        if len(cmd)>10000:
            cmd=cmd.split('\n')
        return cmd
    def _writeHist(self,command):
        # this still brakes for "multiline commands"...
        if self._rechist and command.split(',')[0].strip().lower()=='call':
            cfile=command.split(',')[1].strip().strip('file=').strip('FILE=').strip(';\n').strip('"').strip("'")
            if sys.flags.debug:
                print("DBG: call file ",cfile)
            fin=file(cfile,'r')
            for l in fin:
                self._writeHist(l+'\n')
        else:
            self._hfile.write(command)
    
    def get_sequences(self):
        cdef sequence_list *seqs
        seqs= madextern_get_sequence_list()
        ret={}
        for i in xrange(seqs.curr):
            ret[seqs.sequs[i].name]={'name':seqs.sequs[i].name}
            if seqs.sequs[i].tw_table.name is not NULL:
                ret[seqs.sequs[i].name]['twissname']=seqs.sequs[i].tw_table.name
                print "Table name:",seqs.sequs[i].tw_table.name
                print "Number of columns:",seqs.sequs[i].tw_table.num_cols
                print "Number of columns (orig):",seqs.sequs[i].tw_table.org_cols
                print "Number of rows:",seqs.sequs[i].tw_table.curr
        return ret
        #print "Currently number of sequenses available:",seqs.curr
        #print "Name of list:",seqs.name

def _get_dict(tmpfile,retdict):
    if retdict:
        return tfsDict(tmpfile)
    return tfs(tmpfile)

def _add_range(madrange):
    if madrange:
        if type(madrange)==str:
            return 'range='+madrange+','
        elif type(madrange)==list:
            return 'range='+madrange[0]+'/'+madrange[1]+','
        else:
            raise TypeError("Wrong range type/format")
    else:
        return ''
