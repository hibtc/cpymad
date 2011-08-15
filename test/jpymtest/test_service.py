'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest

from numpy.testing.utils import assert_
from utils import PyMadTestCase
from jpymad.globals import GCont

class Test(unittest.TestCase):


    def testConnect(self):
        pms = PyMadTestCase.pms
        assert_(not GCont.java_gateway == None, 'Must be connected to a java_gateway')
        assert_(not pms.jmad_service == None, 'JMad service must be available')
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testConnect']
    unittest.main()