'''
Created on 15 Aug 2011

@author: kfuchsbe
'''
from pymad.abc import PyMadModelDefinition

class JPyMadModelDefinition(PyMadModelDefinition):
    ''' A wrapper for jmad model-definitions '''
    
    def __init__(self, jmmd):
        self.jmmd = jmmd
    
    @property
    def name(self):
        return self.jmmd.getName()
