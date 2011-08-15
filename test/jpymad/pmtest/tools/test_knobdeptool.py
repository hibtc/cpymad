'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest
from pymad import connect
from pymad.tools import KnobDepTool
from pymad import am 
from pymad import pms

class Test(unittest.TestCase):


    def testKnobDepTool(self):
        connect()
        model = am()
        deltaprange = [-0.001, 0, 0.001]
        
        knobmanager = model.getKnobManager()
        knob = knobmanager.getKnob(pms.enums.KnobType.CUSTOM, 'deltap') #@UndefinedVariable
        modeltool = KnobDepTool(model, knob)
        madxvars = ['x', 'mux', 'y', 'muy']
   
        result = modeltool.calc(madxvars=madxvars, \
                        paramrange=deltaprange)
        print result.names
        print result.s
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testKnobDepTool']
    unittest.main()
