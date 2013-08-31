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
Created on Nov 26, 2010

@author: kaifox
'''
from __future__ import absolute_import

from .tools_optics import get_values
from . import tools_twiss as tw
from cern.pymad.abc import PyMadModel
from .modeldef import JPyMadModelDefinition

class JPyMadModel(PyMadModel):
    '''
    a wrapper for jmad models
    '''
    def __init__(self, jmad_model):
        self.jmm = jmad_model

    @property
    def mdef(self):
        return JPyMadModelDefinition(self.jmm.getModelDefinition())

    @property
    def name(self):
        return self.jmm.getName()

    def set_optic(self, opticname):
        opticdef = self.jmm.getModelDefinition().getOpticsDefinition(opticname)
        if opticdef is None:
            raise ValueError("Optics definition with name '" + opticname + "' can not be found!")

        self.jmm.setActiveOpticsDefinition(opticdef)

    def get_elements(self):
        elements = dict()
        for element in self.jmm.getActiveRange().getElements():
            elements[element.getName()] = element
        return elements

    def get_optics_values(self, madxvarnames):
        optic = self.jmm.getOptics()
        values = dict()
        for name in madxvarnames:
            dict[name] = get_values(optic, name)
        return values

    def twiss(self, seqname=None, columns=[], elementpatterns=['.*'], file=None):
        if not seqname is None:
            print("WARN: seqname='" + seqname + "'. This will be ignored by the jpymad implementation. Instead the active sequence will be used.")

        if not file is None:
            print("WARN: file='" + seqname + "'. This is currently ignored by the jpymad implementation.")

        return tw.twiss(self.jmm, columns, elementpatterns)
