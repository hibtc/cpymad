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

    @abstractproperty
    def seqnames(self):
        ''' Returns a list of the names of the defines sequences in this model definition '''
        pass
    
    @abstractproperty
    def opticnames(self):
        ''' Returns a list of the names of the available optics in this model definition '''
        pass
    
    def __str__(self):
        return self.name
    
    