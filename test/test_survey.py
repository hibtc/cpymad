
import cern.cpymad.api as cpymad
import unittest,os

class TestCpymadSurvey(unittest.TestCase):

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model

    def setUp(self):
        self.model=cpymad.load_model('lhc')

    def test_aperture(self):

        self.assertTrue(hasattr(self,"model"))

        tw1,pw1=self.model.twiss('lhcb1')
        t1,p1=self.model.survey('lhcb1')
        for key in ['angle', 'name', 'l', 's', 'theta', 'y', 'x', 'z']:
            self.assertTrue(key in t1)

if __name__ == '__main__':
    unittest.main()
