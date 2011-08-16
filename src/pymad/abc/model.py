'''
Created on 15 Aug 2011

@author: kfuchsbe
'''
from io import __metaclass__
from abc import ABCMeta, abstractmethod, abstractproperty
class PyMadModel():
    ''' The abstract class for models '''
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def name(self):
        ''' Returns the name of this model '''
        pass
    
    @abstractproperty
    def mdef(self):
        ''' returns the model definition from which the model was created '''
        pass

    @abstractmethod
    def set_optic(self, opticname):
        ''' Sets the actual optic with that of the given name. If the optic is not 
        contained in the definitions, then a ValueError is raised.  
        '''
        pass
    
    @abstractmethod
    def twiss(self, seqname=None, columns=[], elementpatterns=['.*'], file=None):
        ''' Runs a twiss on the model and returns a result containing the variables and the elements given.
        '''
        pass
    
    def __str__(self):
        return self.name
