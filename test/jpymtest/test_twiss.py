'''
Created on Nov 17, 2010

@author: kaifox
'''
import unittest

from utils import PyMadTestCase

class Test(unittest.TestCase):

    def testTwiss(self):
        madxvarnames = ["s", "name", "betx", "bety"]
        result = PyMadTestCase.pms.am.twiss(madxvarnames)
        
        print result

        self.assertTrue(not result["s"] == None, "s-values must be returned")
        try:
            self.assertTrue(result["x"] == None, "x-values were not requested")
        except KeyError:
            pass
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTwiss']
    unittest.main()
