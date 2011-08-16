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
    def mdef(self):
        ''' returns the model definition from which the model was created '''
        pass

    @abstractmethod
    def twiss(self, madxvarnames=[], elementpatterns=['.*']):
        ''' Runs a twiss on the model and returns a result containing the variables and the elements given.
        '''
        pass
