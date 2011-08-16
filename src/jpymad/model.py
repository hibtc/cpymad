'''
Created on Nov 26, 2010

@author: kaifox
'''
from tools_optics import get_values
import tools_twiss as tw
from pymad.abc import PyMadModel
from jpymad.modeldef import JPyMadModelDefinition

class JPyMadModel(PyMadModel):
    '''
    a wrapper for jmad models
    '''
    def __init__(self, jmad_model):
        self.jmm = jmad_model
    
    @property
    def mdef(self):
        return JPyMadModelDefinition(self.jmm.getModelDefinition())
        
    def get_elements(self):
        elements = dict()
        for element in self.jmm.getActiveRange().getElements():
            elements[element.getName()] = element
        return elements
    
    def get_optics_values(self, madxvarnames):
        optic = self.jmm.getOptics()
        values = dict()
        for name in madxvarnames:
            dict[name] = get_values(optic, name)
        return values
    
    def twiss(self, madxvarnames=[], elementpatterns=['.*']):
        return tw.twiss(self.jmm, madxvarnames, elementpatterns)
