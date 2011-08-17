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
Created on Nov 17, 2010

@author: kaifox
'''
import unittest
from jpymad.tools import KnobDepTool
from jpymtest.utils import PyMadTestCase
from jpymad.globals import JPyMadGlobals

class Test(unittest.TestCase):

    def testKnobDepTool(self):
        model =  PyMadTestCase.pms.am
        deltaprange = [-0.001, 0, 0.001]
        
        knobmanager = model.jmm.getKnobManager()
        knob = knobmanager.getKnob(JPyMadGlobals.enums.KnobType.CUSTOM, 'deltap') #@UndefinedVariable
        modeltool = KnobDepTool(model, knob)
        madxvars = ['x', 'mux', 'y', 'muy']
   
        result = modeltool.calc(madxvars=madxvars, \
                        paramrange=deltaprange)
        print result.name
        print result.s
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testKnobDepTool']
    unittest.main()
