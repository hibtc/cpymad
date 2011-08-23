'''
Created on Aug 20, 2011

@author: kaifox
'''
from cern import pymad as pm

#pm.init('cpymad')
#pm.init('jpymad', start='gui')
pms=pm.init('jpymad', start='service')

# print the name of all model definitions
pm.ls_mdefs()

# list the available (running) models
pm.ls_models()

pm.cleanup()
