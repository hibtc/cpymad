'''
Created on 16 Aug 2011

@author: kfuchsbe
'''
from pymad.abc.service import PyMadService
import cpymad,madx

class CpymadService(PyMadService):
    ''' The CPymad implementation of the
        abstract class PyMadService. '''
    
    def mdefs(self):
        return cpymad.modelList
    
    def models(self):
        return madx.list_of_models
    
    def create_model(self, modeldef):
        cpymad.model(modeldef)
    

if __name__=="__main__":
    pmdl=CpymadService.create_model('lhc')
    print pmdl.models()
