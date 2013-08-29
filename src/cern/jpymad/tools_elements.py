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
from globals import JPyMadGlobals

def _iscorrector(element):
    return JPyMadGlobals.java_gateway.jvm.cern.accsoft.steering.jmad.domain.elem.JMadElementType.CORRECTOR.isTypeOf(element) #@UndefinedVariable

def get_kicks(model, hnames, vnames):
    elements = model.get_elements()
    
    hkicks = dict()
    for name in hnames:
        corrector = elements[name]
        if _iscorrector(corrector):
            hkicks[name] = corrector.getKick(JPyMadGlobals.enums.JMadPlane.H) #@UndefinedVariable

    vkicks = dict()
    for name in vnames:
        corrector = elements[name]
        if _iscorrector(corrector):
            vkicks[name] = corrector.getKick(JPyMadGlobals.enums.JMadPlane.V) #@UndefinedVariable
    
    return hkicks, vkicks

def set_kicks(model, hkicks, vkicks):
    '''
    sets the kicks to the given model.
    
    PARAMETERS:
    ===========
    model: the model to which to apply the kicks
    hkicks: a dictionary with element names as keys, horizontal kicks as values
    vkicks: a dictionary with element names as keys, vertical kicks as values
    '''
    elements = model.get_elements()
    
    for name, value in hkicks.items():
        corrector = elements[name]
        if _iscorrector(corrector):
            corrector.setKick(JPyMadGlobals.enums.JMadPlane.H, value) #@UndefinedVariable

    for name, value in vkicks.items():
        corrector = elements[name]
        if _iscorrector(corrector):
            corrector.setKick(JPyMadGlobals.enums.JMadPlane.V, value) #@UndefinedVariable

def add_kicks(model, hkicks, vkicks):
    old_hkicks, old_vkicks = get_kicks(model, hkicks.keys(), vkicks.keys())
    
    new_hkicks = dict()
    for name in hkicks.keys():
        if name in old_hkicks:
            new_hkicks[name] = old_hkicks[name] + hkicks[name]
    
    new_vkicks = dict()
    for name in vkicks.keys():
        if name in old_vkicks:
            new_vkicks[name] = old_vkicks[name] + vkicks[name]
    
    set_kicks(model, new_hkicks, new_vkicks)
    
