'''
Created on 15 Aug 2011

@author: kfuchsbe
'''
from io import __metaclass__
from abc import ABCMeta, abstractproperty

class PyMadModelDefinition():
    '''
    The base class for a model definition
    '''
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def name(self):
        pass

    def __str__(self):
        return self.name
    
    