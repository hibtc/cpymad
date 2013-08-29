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
Created on 16 Aug 2011

.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen[at]cern.ch>
'''
from cern.pymad.abc.service import PyMadService
from cern import cpymad


class CpymadService(PyMadService):
    ''' The CPymad implementation of the
        abstract class PyMadService. '''
    
    def __init__(self, **kwargs):
        self._am=None
        self._models=[]
        for key, value in kwargs.items():
            print "WARN: unhandled option '" + key + "' for CPyMandService. Ignoring it." 
    
    @property
    def mdefs(self):
        return cpymad.modelList()
    
    @property
    def mdefnames(self):
        return self.mdefs
    
    @property
    def models(self):
        mnames=[]
        for m in self._models:
            mnames.append(str(m))
        return mnames
    
    def create_model(self, modeldef):
        self._models.append(cpymad.model(modeldef))
        self._am=self._models[-1]
        return self._am
    
    def am(self):
        return self._am
    
    def delete_model(self,model):
        '''
         The cpymad implementation of this is to simply
         remove all references to the model. If the reference count is
         still different from zero, it will be kept in memory.
        '''
        if str(self._am)==model:
            self._am=None
        for i in range(len(self._models)):
            if model==str(self._models[i]):
                del self._models[i]

