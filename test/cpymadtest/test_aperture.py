from cern import cpymad
import unittest,os

class TestCpymadAperture(unittest.TestCase):

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model
    
    def test_aperture(self):
        '''
         Tests a specific aperture from LHC model
        '''
        self.model=cpymad.model('lhc')
        aper,pars=self.model.aperture(madrange='ir2')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCpymadAperture)
    unittest.TextTestRunner(verbosity=1).run(suite)
