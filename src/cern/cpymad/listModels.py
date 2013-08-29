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

modelpaths = [os.path.join(os.path.dirname(__file__),'_models')]

def modelList():
    return _get_mnames_files()[0]

def _get_mnames_files():
    """List availabel models and corresponding file pathes.

    Searches for all .cpymad.json files within all `modelpaths` folders.
    Returns a list [model names] and a dictionary {file => [model names]}.
    No care is taken to prevent a model from being listed multiple times.
    """
    if USE_COUCH:
        return cern.cpymad._couch_server.ls_models()
    mnames=[]
    fnames={}
    for modelloc in modelpaths:
        for f in os.listdir(modelloc):
            if not f.lower().endswith('.cpymad.json'):
                continue
            path = os.path.join(modelloc, f)
            fnames[path]=[]
            jloaded=json.load(file(path))
            for mname in jloaded.keys():
                if jloaded[mname]['real']:
                    mnames.append(mname)
                    fnames[path].append(mname)
    return mnames,fnames


if __name__=="__main__":
    print modelList()
