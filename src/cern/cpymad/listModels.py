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
import os,json
from cern.pymad.globals import USE_COUCH
import cern.cpymad

def modelList():
    if USE_COUCH:
        return cern.cpymad._couch_server.ls_models()
    pymadloc=os.path.dirname(__file__)
    modelloc=os.path.join(pymadloc,'_models')
    ret=[]
    for f in os.listdir(modelloc):
        if len(f)>5 and f[-12:].lower()=='.cpymad.json':
            for mname in json.load(file(os.path.join(modelloc,f))).keys():
                ret.append(mname)
            
    print "Models:",ret
    return ret


if __name__=="__main__":
    print modelList()