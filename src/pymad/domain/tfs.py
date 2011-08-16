'''
Created on 16 Aug 2011

@author: kfuchsbe
'''
from string import lower


class LookupDict():
    ''' A dictionary like structur, which exposes the values of the keys also as attributes with the key names '''
    
    def __init__(self, values):
        ''' Initializes the class with the values.
        
        Parameters:
        values -- A dictionary with strings as keys and lists as values
        '''
        # we store the values in a new dict internally, to unify the keys to lowercase
        self._values = dict()
        for key, val in values.items():
            self._values[self._unify_key(key)] = val

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
