'''
Created on Aug 20, 2011

@author: kaifox
'''
import pymad as pm

#pm.init('cpymad')
pm.init('jpymad', start='gui')

# print the name of all model definitions
pm.ls_mdefs()

# list the available (running) models
pm.ls_models()

pm.cleanup()