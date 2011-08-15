'''
Created on Mar 22, 2011

@author: kaifox
'''
from jpymad.service import JPyMadService

class PyMadTestCase(): 
    
    pms = JPyMadService()
    model = pms.create_model("TI2")
    pms.set_am(model) 
    
    
