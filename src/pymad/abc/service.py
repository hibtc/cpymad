'''
Created on 15 Aug 2011

@author: kfuchsbe
'''
from io import __metaclass__
from abc import ABCMeta, abstractmethod, abstractproperty

class PyMadService():
    ''' The abstract class for a model-service. '''
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def mdefs(self):
        ''' Returns all the available model definitions as a list '''
        pass
    
    @abstractproperty
    def mdefnames(self):
        ''' Returns all the names of the available model definitions '''
        pass
        
    @abstractproperty
    def models(self):
        ''' Returns all the instantiated models as a list '''
        pass
    
    @abstractproperty
    def am(self):
        ''' Returns the active model '''
        pass
    
    @abstractmethod
    def create_model(self, modeldef):
        """Create a model instance from a model definition.

        Arguments:
        modeldef -- the model definition from which to create the model as an 
                    object or just its name. If a name is given, then the modeldefinition 
                    is first searched in the available model definitions.
        
        """
        pass
    
    @abstractmethod
    def delete_model(self, model):
        """ cleans up the given model and deletes a model from the available models """
        pass
    
    def cleanup(self):
        """ Can be overridden by subclass to do necessary cleanup steps """
        pass
