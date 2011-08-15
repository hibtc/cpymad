include "pymadx/table.pyx"
cdef extern from "madextern.h":
    void madextern_start()
    void madextern_end()
    void madextern_input(char*)
    #table *getTable()

import os,sys
from pymadx import tfs


##
# @brief Python class which interfaces to Mad-X
# Linking to the C Library.
class madx:
    
    ##
    # Initialize object
    def __init__(self,histfile='',recursive_history=False):
        madextern_start()
        if histfile:
            self._hist=True
            self._hfile=file(histfile,'w')
            self._rechist=recursive_history
        else:
            self._hist=False
            if recursive_history:
                print("WARNING: you cannot get recursive history without history file...")
            self._rechist=False
    
    ##
    # Closes history file
    # and calls the madextern_end()
    # function.
    def __del__(self):
        if self._rechist:
            self._hfile.close()
        madextern_end()
    ##
    # @brief debug command to attempt to figure out table struct
    def debug(self):
        cdef table tab
        tab=table(15)
        print tab.keys()
        #print tab.num_cols
    
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
        cmd=self._fixcmd(cmd)
        if type(cmd)==int: # means we should not execute command
            return cmd
        if self._checkCommand(cmd.lower()):
            madextern_input(cmd)
        if self._hist:
            if cmd[-1]=='\n':
                self._writeHist(cmd)
            else:
                self._writeHist(cmd+'\n')
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
    # @param columns [string] the columns you want for the flag
    def select(self,flag,pattern,columns):
        self.command('SELECT, FLAG='+flag+', CLEAR;')
        self.command('SELECT, FLAG='+flag+', PATTERN='+pattern[0]+', COLUMN='+columns+';')
        for i in range(1,len(pattern)):
            self.command('SELECT, FLAG='+flag+', PATTERN='+pattern[i]+';')
            
    ##
    # @brief Runs select+use+twiss on the sequence selected
    # 
    # @param sequence [string] name of sequence
    # @param fname [string,optional] name of file to store tfs table
    # @param pattern [list, optional] pattern to include in table
    # @param columns [string, optional] columns to include in table
    def twiss(self,
              sequence,
              pattern=['full'],
              columns='name,s,betx,bety,x,y,dx,dy,px,py,mux,muy',
              fname=''
              ):
        if type(sequence)!=str:
            print("ERROR: sequence must be a string")
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
        self.command('use, sequence='+sequence+';')
        self.command('twiss, sequence='+sequence+', file="'+tmpfile+'";')
        tab,param=tfs(tmpfile)
        if not fname:
            os.remove(tmpfile)
        return tab,param
            
    ##
    # @brief Runs select+use+survey on the sequence selected
    # 
    # @param sequence [string] name of sequence
    # @param fname [string,optional] name of file to store tfs table
    # @param pattern [list, optional] pattern to include in table
    # @param columns [string, optional] columns to include in table
    def survey(self,
              sequence,
              pattern=['full'],
              columns='name,l,angle,x,y,z,theta',
              fname=''
              ):
        if type(sequence)!=str:
            print("ERROR: sequence must be a string")
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
        self.command('use, period='+sequence+';')
        self.command('survey, file="'+tmpfile+'";')
        tab,param=tfs(tmpfile)
        if not fname:
            os.remove(tmpfile)
        return tab,param
    
    # turn on/off verbose outupt..
    def verbose(self,switch):
        if switch:
            self.command("OPTION, ECHO, WARN, INFO;")
        else:
            self.command("OPTION, -ECHO, -WARN, -INFO;")

    def _fixcmd(self,cmd):
        if type(cmd)!=str and type(cmd)!=unicode:
            print("ERROR: input must be a string")
            return 1
        if len(cmd.strip())==0:
            return 0
        if cmd.strip()[-1]!=';':
            cmd+=';'
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
