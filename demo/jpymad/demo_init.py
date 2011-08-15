'''
Created on Nov 11, 2010

@author: kaifox
'''

from jpymad.service import JPyMadService

pms = JPyMadService()

mdefs = pms.get_mdefs()

for mdef in mdefs:
    print mdef
    

