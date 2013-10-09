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
Created on Nov 17, 2010

@author: kaifox
'''

from conversions import tofl
from conversions import tostr
from globals import JPyMadGlobals
from cern.pymad.domain.tfs import TfsTable, TfsSummary

def twiss(model, madxvarnames, elementpatterns=['.*']):

    # create the request
    madxvars = []
    outnames = []
    for name in madxvarnames:
        var = JPyMadGlobals.enums.MadxTwissVariable.fromMadxName(name) #@UndefinedVariable
        if var is not None:
            madxvars.append(var)
            outnames.append(name)

    tfsResultRequest = JPyMadGlobals.java_gateway.jvm.cern.accsoft.steering.jmad.domain.result.tfs.TfsResultRequestImpl() #@UndefinedVariable

    for pattern in elementpatterns:
        tfsResultRequest.addElementFilter(pattern)

    for var in madxvars:
        tfsResultRequest.addVariable(var)

    # do the twiss
    tfsResult = model.twiss(tfsResultRequest)

    results = dict()
    for idx, var in enumerate(madxvars):
        vartype = tfsResult.getVarType(var)
        if vartype == JPyMadGlobals.enums.MadxVarType.STRING: #@UndefinedVariable
            values = tostr(tfsResult.getStringData(var))
        else:
            values = tofl(tfsResult.getDoubleData(var))
        results[outnames[idx]] = values

    params = dict()
    tfsSummary = tfsResult.getSummary()
    for key in tfsSummary.getKeys():
        vartype = tfsSummary.getVarType(key)
        if vartype == JPyMadGlobals.enums.MadxVarType.STRING: #@UndefinedVariable
            value = str(tfsSummary.getStringValue(key))
        else:
            value = float(tfsSummary.getDoubleValue(key))
        params[key] = value


    return TfsTable(results), TfsSummary(params)

