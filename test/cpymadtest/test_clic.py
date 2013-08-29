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
import unittest
from cern import cpymad
from base_test import TestCpymad

class TestCLIC(TestCpymad):
    def setUp(self):
        self.model=cpymad.model('clic')
        self.model._cmd('option,-twiss_print')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCLIC)
    unittest.TextTestRunner(verbosity=1).run(suite)

