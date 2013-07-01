
from cern import cpymad
import unittest,os

class TestCpymadSurvey(unittest.TestCase):

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model
    
    def test_aperture(self):

        self.model=cpymad.model('lhc')

        sur_c='name,s,l,x,y,z,theta'

        tw1,pw1=self.model.twiss('lhcb1')
        t1,p1=self.model.survey('lhcb1')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCpymadSurvey)
    unittest.TextTestRunner(verbosity=1).run(suite)
