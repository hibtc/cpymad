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
    
    @property
    def name(self):
        return self.jmm.getName()
    
    def set_optic(self, opticname):
        opticdef = self.jmm.getModelDefinition().getOpticsDefinition(opticname)
        if opticdef is None:
            raise(ValueError("Optics definition with name '" + opticname + "' can not be found!"));
        
        self.jmm.setActiveOpticsDefinition(opticdef)
    
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
    
    def twiss(self, seqname=None, columns=[], elementpatterns=['.*'], file=None):
        if not seqname is None:
            print("WARN: seqname='" + seqname + "'. This will be ignored by the jpymad implementation. Instead the active sequence will be used.")
        
        if not file is None:
            print("WARN: file='" + seqname + "'. This is currently ignored by the jpymad implementation.")
        
        return tw.twiss(self.jmm, columns, elementpatterns)
