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
contains convenience funtions to initialize the pymad envirenment and to clean up

Created on 16 Aug 2011

@author: kfuchsbe
'''

from cern.pymad.globals import PyMadGlobals

class PyMadNotInititalizedError(Exception):
    pass

def get_pms():
    ''' Returns the pymad-service singleton. If pymad is not initialized this raises an error '''
    if not is_initialized():
        raise(PyMadNotInititalizedError())

    return PyMadGlobals.PYMAD_SERVICE()

def is_initialized():
    ''' returns true, if a pymad service is initialized, false otherwise '''
    return (not (PyMadGlobals.PYMAD_SERVICE is None or PyMadGlobals.PYMAD_SERVICE() is None) )



def init(mode='cpymad', **kwargs):
    ''' Initializes the environment for using pymad in one of the possible modes. The first argument determines
    the mode (allowed values are 'cpymad' and 'jpymad'. If already a pymad service is running, then this function
    prints a warning and does nothing else.

    :param string mode: Can be either 'cpymad' or 'pymad'. This option determines which pymad-service will be initialized.
    :param kwargs: All the rest of the arguments are directly passed to the constructor of the created PyMadService
    :returns: the newly created pymad service if one was created.

    '''

    if is_initialized():
        print("Already a pymad service running. Doing nothing.")
        return None
    if mode is 'cpymad':
        from cern.cpymad.service import CpymadService
        pms = CpymadService(**kwargs)
    elif mode is 'jpymad':
        from cern.jpymad.service import JPyMadService
        pms = JPyMadService(**kwargs)
    else:
        raise ValueError("Unknown mode '" + mode + "'! Use one of 'cpymad' or 'jpymad'!")
    import weakref
    PyMadGlobals.PYMAD_SERVICE = weakref.ref(pms)
    return pms

def cleanup():
    ''' Cleans up the pymad service and sets the global variable back to None again. '''

    if not is_initialized():
        print "pymad is not initialized. Doing nothing."
        return

    pms = PyMadGlobals.PYMAD_SERVICE()
    pms.cleanup()



