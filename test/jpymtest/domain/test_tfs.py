'''
Created on 16 Aug 2011

@author: kfuchsbe
'''
import unittest
from pymad.domain.tfs import TfsTable


class Test(unittest.TestCase):

    def setUp(self):
        values = dict(name=["A", "A.B", "c"], x=[0.1, 0.3, 0.5], betx=[2.1, 2.2, 2.3])
        self.read_tfs = TfsTable(values)

    def testGetX(self):
        self.assertEquals([0.1, 0.3, 0.5], self.read_tfs['x'])
        self.assertEquals([0.1, 0.3, 0.5], self.read_tfs['X'])
        self.assertEquals([0.1, 0.3, 0.5], self.read_tfs.x)
        self.assertEquals([0.1, 0.3, 0.5], self.read_tfs.X)


    def testGetNames(self):
        self.assertEquals(["A", "A.B", "c"], self.read_tfs['name'])
        self.assertEquals(["A", "A.B", "c"], self.read_tfs['NAME'])
        self.assertEquals(["A", "A.B", "c"], self.read_tfs.name)
        self.assertEquals(["A", "A.B", "c"], self.read_tfs.NaMe)
        self.assertEquals(["A", "A.B", "c"], self.read_tfs.names)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGetX']
    unittest.main()