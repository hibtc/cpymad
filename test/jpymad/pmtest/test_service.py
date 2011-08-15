'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest

from pymad.service import pms 
from pymad.service import connect
from numpy.testing.utils import assert_

class Test(unittest.TestCase):


    def testConnect(self):
        connect()
        assert_(not pms.java_gateway == None, 'Must be connected to a java_gateway')
        assert_(not pms.jmad_service == None, 'JMad service must be available')
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testConnect']
    unittest.main()