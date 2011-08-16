'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest

from utils import PyMadTestCase
from jpymad.globals import JPyMadGlobals

class Test(unittest.TestCase):


    def testConnect(self):
        pms = PyMadTestCase.pms
        self.assertTrue(not JPyMadGlobals.java_gateway is None, 'Must be connected to a java_gateway')
        self.assertTrue(not pms.jmad_service is None, 'JMad service must be available')
        for mdef in pms.mdefs:
            print mdef


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testConnect']
    unittest.main()