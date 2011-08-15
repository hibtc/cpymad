'''
Created on Mar 22, 2011

@author: kaifox
'''
import unittest
import pymad as pm

class PyMadTestCase(unittest.TestCase):
    pm.connect()
    model = pm.create_model("TI2")
    pm.set_am(model) 
    
    
