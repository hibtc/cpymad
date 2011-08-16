'''
Created on 16 Aug 2011

@author: kfuchsbe
'''
import unittest
from jpymad.service import JPyMadService


class Test(unittest.TestCase):

    def setUp(self):
        self.pms = JPyMadService()
        self.mdef = self.pms.get_mdef("TI2")
        
    def testGetName(self):
        self.assertTrue(not self.mdef is None)
        name = self.mdef.name
        self.assertEquals("TI2", name)
    
    def testGetSequenceNames(self):
        seqnames = self.mdef.seqnames
        self.assertEquals(1, len(seqnames))
        self.assertEquals("ti2", seqnames[0])
        
    def testGetOpticNames(self):
        opticnames = self.mdef.opticnames
        self.assertEquals(1, len(opticnames))
        self.assertEquals('default optics', opticnames[0])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGetName']
    unittest.main()
