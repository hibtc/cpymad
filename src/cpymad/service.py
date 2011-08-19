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

@author: kfuchsbe
'''
from pymad.abc.service import PyMadService
import cpymad,madx

class CpymadService(PyMadService):
    ''' The CPymad implementation of the
        abstract class PyMadService. '''
    
    def __init__(self):
        self._am=None
        self.madx=madx.madx()
    
    
    def mdefs(self):
        return cpymad.modelList()
    
    def mdefnames(self):
        return self.mdefs()
    
    def models(self):
        return self.madx.list_of_models()
    
    def create_model(self, modeldef):
        self._am=cpymad.model(modeldef)
        return self._am
    
    def am(self):
        return self._am
    def delete_model(self):
        del self._am
        self._am=None
    

if __name__=="__main__":
    pmdl=CpymadService.create_model('lhc')
    print pmdl.models()
