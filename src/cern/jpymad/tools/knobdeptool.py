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
import numpy as np

class KnobDepTool():
    '''
    This class provides some simple functions to calc parameter
    - dependencies of the model
    '''
    def __init__(self, model, knob):
        self._model = model
        self._knob = knob

    def calc(self, madxvars=['x', 'mux'], \
        paramrange=np.linspace(-0.002, 0.002, 21), \
        elementpatterns=['.*']):
        '''
        this method calculates the dependency of the given variable
        on the given knob
        '''

        oldvalue = self._knob.getOffset()

        columns = list(madxvars)
        columns.append('name')
        columns.append('s')

        data = None

        for paramvalue in paramrange:
            self._knob.setOffset(float(paramvalue))
            twresult, params = self._model.twiss(columns, elementpatterns)

            if data is None:
                data = Data()
                data.name = np.array(twresult['name'])
                data.s = np.array(twresult['s'])
                data.paramrange = paramrange
                for madxvar in madxvars:
                    #print madxvar + ": len=" + len(twresult[madxvar])
                    setattr(data, madxvar, np.array([twresult[madxvar]]))
            else:
                for madxvar in madxvars:
                    #print madxvar + ": len=" + len(twresult[madxvar])
                    setattr(data, madxvar, np.append(getattr(data, madxvar), \
                                     np.array([twresult[madxvar]]), axis=0))

        self._knob.setOffset(oldvalue)

        return data

class Data(object):
    pass
