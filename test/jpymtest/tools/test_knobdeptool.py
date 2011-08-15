'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest
from jpymad.tools import KnobDepTool
from jpymtest.utils import PyMadTestCase
from jpymad.globals import GCont

class Test(unittest.TestCase):

    def testKnobDepTool(self):
        model =  PyMadTestCase.pms.am()
        deltaprange = [-0.001, 0, 0.001]
        
        knobmanager = model.jmm.getKnobManager()
        knob = knobmanager.getKnob(GCont.enums.KnobType.CUSTOM, 'deltap') #@UndefinedVariable
        modeltool = KnobDepTool(model, knob)
        madxvars = ['x', 'mux', 'y', 'muy']
   
        result = modeltool.calc(madxvars=madxvars, \
                        paramrange=deltaprange)
        print result.name
        print result.s
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testKnobDepTool']
    unittest.main()
