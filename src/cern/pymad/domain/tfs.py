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
.. module:: tfs
.. moduleauthor:: kfuchsbe
'''
from string import lower


class LookupDict():
    ''' A dictionary-like structure, which exposes the values of the keys also as attributes with the key names '''
    
    def __init__(self, values):
        ''' Initializes the class with the values.
        
        Parameters:
        values -- A dictionary with strings as keys and lists as values
        '''
        # we store the values in a new dict internally, to unify the keys to lowercase
        self._values = dict()
        for key, val in values.items():
            self._values[self._unify_key(key)] = val
    
    def __iter__(self):
        return iter(self._values)
        
    def _get_val_or_raise_error(self, key, error):
        ukey = self._unify_key(key)
        
        if (self._values.has_key(ukey)):
            return self._values[key]
        else:
            raise(error)
        
    def __getattr__(self, name):
        ''' Exposes the variables as attributes. This allows to use code like the following:
        
        tfs = TfsTable(...)
        print tfs.x
         
        '''
        return self._get_val_or_raise_error(name, AttributeError())
            
    def __getitem__(self, key):
        ''' Emulates the [] operator, so that TfsTable can be used just like a dictionary. 
            The keys are considered case insensitive.
        '''
        return self._get_val_or_raise_error(key, KeyError())
        
    def _unify_key(self, key):
        return lower(key)
    
    def keys(self):
        '''
         Similar to dictionary.keys()...
        '''
        return self._values.keys()


class TfsTable(LookupDict):
    ''' A class to hold the results of a twiss '''
    def __init__(self, values):
        LookupDict.__init__(self, values)
        if self._values.has_key('name'):
            self._names = self._values['name']
    
    @property
    def names(self):
        ''' Returns the names of the elements in the twiss table '''
        return self._names

class TfsSummary(LookupDict):
    ''' A class to hold the summary table of a twiss with lowercase keys '''
    pass
