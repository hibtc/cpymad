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
Created on 15 Aug 2011

@author: kfuchsbe
'''
from __future__ import absolute_import

from abc import abstractproperty

from .interface import Interface

class PyMadModelDefinition(Interface):
    '''
    The base class for a model definition
    '''

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def seqnames(self):
        ''' Returns a list of the names of the defined sequences in this model definition '''
        pass

    @abstractproperty
    def opticnames(self):
        ''' Returns a list of the names of the available optics in this model definition '''
        pass

    def __str__(self):
        return self.name


