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
Created on 16 Aug 2011

@author: kfuchsbe
'''
class PyMadGlobals():
    ''' A class containing only the singletons used for the pymad convenience functions. 
    
    This class shall never be instantiated.
    '''
    
    # the pymad-service singleton
    PYMAD_SERVICE=None
    
'''
 Use couchdb (only inside CERN at the moment..)
 Does not work yet, problems with
 multiprocessing and accesses to the server.
'''
USE_COUCH=False

'''
 If this is a string (i.e. "if MAD_HISTORY_BASE" is True),
 then every madx instance will write a history
 file with this as base name.
 an integer will be appended, pluss file ending '.madx'
 If you in your script specify histfile, that takes precedence
'''
MAD_HISTORY_BASE=''