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
'''
Created on Sep 1, 2009

calculates the mean dispersion at the MBIs

@author: kfuchsbe
'''

import numpy as np
from jpymad.service import JPyMadService

pms = None

def ensure_model(pms):
    '''
    ensures, that the desired model is created and active
    '''
    active_model = pms.am()
    if active_model is None:
        new_model = pms.create_model("TI8")
        pms.set_am(new_model)
    
def calc_mean_dispersion(model,elementpatterns=['MBI[T\.].*']):
    """
    calculates the mean dispersion at the elements, which match the given regex
    """
    tw = model.twiss(['name','dx'], elementpatterns)
    print 'elements:'
    print '------'
    print tw['name']
    print 'mean of dispersion on elements=' + str(np.average(tw['dx']))


if __name__ == "__main__":
    pms = JPyMadService()
    ensure_model(pms)
    model = pms.am()
    calc_mean_dispersion(model)
