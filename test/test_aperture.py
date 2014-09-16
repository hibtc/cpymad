import cern.cpymad.api as cpymad
import unittest

class TestCpymadAperture(unittest.TestCase):

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model

    def test_aperture(self):
        '''
         Tests a specific aperture from LHC model
        '''
        self.model=cpymad.load_model('lhc')
        aper,pars=self.model.aperture(madrange='ir2')


if __name__ == '__main__':
    unittest.main()
