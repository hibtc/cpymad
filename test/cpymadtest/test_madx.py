import unittest
from cern.madx import madx
from math import pi

class TestMadX(unittest.TestCase):
    """Test methods of the madx class."""

    def setUp(self):
        self.madx = madx()

    def tearDown(self):
        del self.madx

    def testEvaluate(self):
        self.madx.command("FOO = PI*3;")
        val = self.madx.evaluate("1/FOO")
        self.assertAlmostEqual(val, 1/(3*pi))

if __name__ == '__main__':
    unittest.main()
