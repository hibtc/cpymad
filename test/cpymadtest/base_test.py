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
import unittest,os


class TestCpymad(unittest.TestCase):

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model

    def test_twiss(self):
        t,p=self.model.twiss()
        for attr in ['betx','bety','s']:
            self.assertTrue(hasattr(t,attr))
        # check that keys are all lowercase..
        for k in t:
            self.assertTrue(k==k.lower())
        for k in p:
            self.assertTrue(k==k.lower())

    def test_sequences(self):
        '''
         Checks that all sequences defined in the model (json)
         is also loaded into memory
        '''
        for seq in self.model.mdef['sequences'].keys():
            print "Testing sequence",seq
            self.assertTrue(self.model.has_sequence(seq))

    def test_set_optic(self):
        '''
         Sets all optics found in the model definition
        '''
        for optic in self.model.list_optics():
            print "Testing optics",optic
            self.model.set_optic(optic)
            self.assertEqual(optic,self.model._active['optic'])
            self.model.twiss()

