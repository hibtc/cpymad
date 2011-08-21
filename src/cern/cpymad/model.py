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


class model:
    '''
    model class implementation.
     
    :param string model: Name of model to load.
    :param string optics: Name of optics to load
    :param string history: Name of file which will contain all Mad-X commands called.
    '''
    def __init__(self,model,optics='',history=''):
        self.madx=madx(histfile=history)
        
        already_loaded=False
        for m in self.madx.list_of_models():
            if model==m.model:
                _deepcopy(m,self)
                already_loaded=True
        
        if not already_loaded:
            # check that the model does not conflict with existing models..
            _check_compatible(model)
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
            
            
            self.madx.verbose(False)
            self.madx.command(_get_file_content(os.path.join('_models',self._dict['header'])))
            
            self._call(self._dict['sequence'])
            
            self._optics=''
            self.madx.append_model(self)
        
        self.set_optics(optics)
    
    def __del__(self):
        del self.madx
    
    def __str__(self):
        return self.model
    
    def _call(self,f):
        self.madx.call(self._db+f)
    
    def has_sequence(self,sequence):
        '''
         Check if model has the sequence.
         
         :param string sequence: Sequence name to be checked.
        '''
        return sequence in self.madx.get_sequences()
    
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
                self.madx.command(e+":="+val)
        
        for b in odict['beams']:
            self.madx.command(bdict[b])
        self._optics=optics
    
    def list_sequences(self):
        return self.madx.get_sequences()
    
    def list_optics(self):
        return self._dict['optics'].keys()
    
    def twiss(self,sequence="",columns=""):
        if sequence=="":
            sequence=self._dict['default']['sequence']
        if columns:
            return self.madx.twiss(sequence=sequence,columns=columns)
        else:
            return self.madx.twiss(sequence=sequence)
            


def _deepcopy(origin,new):
    new.madx=origin.madx
    new.model=origin.model
    new._dict=origin._dict
    new._db=origin._db
    new._optics=origin._optics
  
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
    


def _check_compatible(model):
    m=madx()
    d=_get_data(model)
    if len(m.list_of_models())>0:
        # we always throw error until we know this actually can work..
        raise ValueError("Two models cannot be loaded at once at this moment")
