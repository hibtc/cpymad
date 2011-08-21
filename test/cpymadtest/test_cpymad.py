#-------------------------------------------------------------------------------
# This file is part of PyMad.
# 
# Copyright (c) 2011, CERN. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# 	http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
from cern import cpymad
import unittest


class TestCpymad(unittest.TestCase):
    
    def setUp(self):
        self.lhc=cpymad.model('lhc',history='hist_cpymad.madx')
        
    def test_twiss(self):
        t,p=self.lhc.twiss()
        for attr in ['betx','bety','s']:
            self.assertTrue(hasattr(t,attr))
        # check that keys are all lowercase..
        for k in t:
            self.assertTrue(k==k.lower())
        for k in p:
            self.assertTrue(k==k.lower())
        
    def test_sequences(self):
        self.assertFalse(self.lhc.has_sequence('non_existing_sequence'))
        for seq in ['lhcb1','lhcb2']:
            self.assertTrue(self.lhc.has_sequence(seq))
    
    def test_wrong_optics(self):
        with self.assertRaises(KeyError):
            self.lhc.set_optics('non_existing_optics')
    
    def test_has_optics(self):
        self.assertFalse(self.lhc.has_optics('non_existing_optics'))
        self.assertTrue(self.lhc.has_optics('collision'))
    
    def test_listModels(self):
        l=cpymad.modelList()
        self.assertTrue('lhc' in l)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCpymad)
    unittest.TextTestRunner(verbosity=1).run(suite)
