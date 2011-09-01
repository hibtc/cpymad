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
.. module: cern.abc.pymad.model

Created on 15 Aug 2011

.. moduleauthor:: Kajetan Fuchsberger <Kajetan.Fuchsberger@cern.ch>

'''
from abc import ABCMeta, abstractmethod, abstractproperty
class PyMadModel():
    ''' The abstract class for models '''
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def name(self):
        ''' Returns the name of this model '''
        pass
    
    @abstractproperty
    def mdef(self):
        ''' returns the model definition from which the model was created '''
        pass

    @abstractmethod
    def set_optic(self, opticname):
        ''' Sets the actual optic with that of the given name. If the optic is not 
        contained in the definitions, then a ValueError is raised.  
        '''
        pass
    
    @abstractmethod
    def twiss(self, seqname=None, columns=[], elementpatterns=['.*'], file=None):
        ''' Runs a twiss on the model and returns a result containing the variables and the elements given.
        '''
        pass
    
    def __str__(self):
        return self.name
