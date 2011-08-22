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
.. module: cpymad.model

Cython implementation of the model api.
:See also: :mod:`pymad.abc.model`

.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>

'''

import json, os, sys
from cern.madx import madx
#from cern.pymad import abc.model
import multiprocessing


#class model(model.PyMadModel):
class model():
    '''
    model class implementation. the model spawns a madx instance in a separate process.
    this has the advantage that you can run separate models which do not affect each other.
     
    :param string model: Name of model to load.
    :param string optics: Name of optics to load
    :param string history: Name of file which will contain all Mad-X commands called.
    '''
    def __init__(self,model,optics='',history=''):
        
        # name of model:
        self.model=model
        # loading the dictionary...
        self._dict=_get_data(model)
        
        self._db=None
        for db in self._dict['dbdir']:
            if os.path.isdir(db):
                self._db=db
        if not self._db:
            raise ValueError,"Could not find an available database path"
        
        # Defining two pipes which are used for communicating...
        _child_pipe_recv,_parent_send=multiprocessing.Pipe(False)
        _parent_recv,_child_pipe_send=multiprocessing.Pipe(False)
        self._send=_parent_send.send
        self._recv=_parent_recv.recv
        
        self._mprocess=multiprocessing.Process(target=_modelProcess, 
                                               args=(_child_pipe_send,_child_pipe_recv,model,history))
        self._mprocess.start()
        
        self._call(self._dict['sequence'])
        
        self._optics=''
        
        self.set_optics(optics)
    
    def __del__(self):
        print "Trying to delete model..."
        try:
            self._send('delete_model')
            self._mprocess.join(5)
        except TypeError:
            pass
    def __str__(self):
        return self.model
    
    def _call(self,f):
        self._send(('call',self._db+f))
        return self._recv()
        #self.madx.call(self._db+f)
    
    def has_sequence(self,sequence):
        '''
         Check if model has the sequence.
         
         :param string sequence: Sequence name to be checked.
        '''
        return self._sendrecv(('has_sequence',sequence))
        #return sequence in self.madx.get_sequences()
    
    def has_optics(self,optics):
        '''
         Check if model has the optics.
         
         :param string optics: Optics name to be checked.
        '''
        return optics in self._dict['optics']
    
    def set_optics(self,optics):
        '''
         Set new optics.
         
         :param string optics: Optics name.
         
         :raises KeyError: In case you try to set an optics not available in model.
        '''
        if optics=='':
            optics=self._dict['default']['optics']
        if self._optics==optics:
            print("INFO: Optics already initialized")
            return 0
        
        # optics dictionary..
        odict=self._dict['optics'][optics]
        # flags dictionary..
        fdict=self._dict['flags']
        bdict=self._dict['beams']
        
        for strfile in odict['strengths']:
            self._call(strfile)
        
        for f in odict['flags']:
            val=odict['flags'][f]
            for e in fdict[f]:
                self._cmd(e+":="+val)
                #self.madx.command(e+":="+val)
        
        for b in odict['beams']:
            self._cmd(bdict[b])
            #self.madx.command(bdict[b])
        self._optics=optics
    
    def list_sequences(self):
        return self._sendrecv('get_sequences')
    
    def list_optics(self):
        return self._dict['optics'].keys()
    
    def twiss(self,sequence="",columns=""):
        from cern.pymad.domain import TfsTable, TfsSummary
        if sequence=="":
            sequence=self._dict['default']['sequence']
        t,s=self._sendrecv(('twiss',sequence,columns))
        return TfsTable(t),TfsSummary(s)
        #return self.madx.twiss(sequence=sequence,columns=columns)
    
    def _cmd(self,command):
        self._send(('command',command))
        return self._recv()
    def _sendrecv(self,func):
        self._send(func)
        return self._recv()
        
           
def _get_data(modelname):
    fname=os.path.join('_models',modelname+'.json')
    _dict = _get_file_content(fname)
    return json.loads(_dict)

def _get_file_content(filename):
    try:
         import pkgutil
         stream = pkgutil.get_data(__name__, filename)
    except ImportError:
        import pkg_resources
        stream = pkg_resources.resource_string(__name__, filename)
    return stream
    


def _modelProcess(sender,receiver,model,history=''):
    _madx=madx(histfile=history)
    _dict=_get_data(model)
    _madx.verbose(False)
    _madx.append_model(model)
    _madx.command(_get_file_content(os.path.join('_models',_dict['header'])))
    while True:
        cmd=receiver.recv()
        if cmd=='delete_model':
            sender.close()
            receiver.close()
            break
        elif cmd[0]=='call':
            _madx.call(cmd[1])
            sender.send('done')
        elif cmd[0]=='has_sequence':
            sender.send(cmd[1] in _madx.get_sequences())
        elif cmd[0]=='command':
            _madx.command(cmd[1])
            sender.send('done')
        elif cmd[0]=='twiss':
            if cmd[2]:
                t,p=_madx.twiss(sequence=cmd[1],columns=cmd[2],retdict=True)
            else:
                t,p=_madx.twiss(sequence=cmd[1],retdict=True)
            sender.send((t,p))
        else:
            raise ValueError("You sent a wrong command to subprocess")
            
