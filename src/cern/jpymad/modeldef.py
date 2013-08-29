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
from cern.pymad.abc import PyMadModelDefinition

class JPyMadModelDefinition(PyMadModelDefinition):
    ''' A wrapper for jmad model-definitions '''

    def __init__(self, jmmd):
        self.jmmd = jmmd

    @property
    def name(self):
        return self.jmmd.getName()

    @property
    def seqnames(self):
        names = []
        for sequence in self.jmmd.getSequenceDefinitions():
            names.append(sequence.getName());
        return names

    @property
    def opticnames(self):
        names = []
        for optic in self.jmmd.getOpticsDefinitions():
            names.append(optic.getName());
        return names
