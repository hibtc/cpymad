'''
Created on Nov 17, 2010

@author: kaifox
'''
import pmtest.utils as tu 
import unittest

from pymad import am
from pymad import connect

class Test(tu.PyMadTestCase):

    def testTwiss(self):
        connect()
        madxvarnames = ["s", "name", "betx", "bety"]
        result = am().twiss(madxvarnames)
        
        print result

        self.assertTrue(not result["s"] == None, "s-values must be returned")
        try:
            self.assertTrue(result["x"] == None, "x-values were not requested")
        except KeyError:
            pass
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testTwiss']
    unittest.main()
