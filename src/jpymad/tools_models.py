# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:21:25 2010

@author: kaifox
"""

def ls_mdefs(pms):
    """
    lists all the currently available model definitions from the given service
    
    Arguments:
    pms -- the pymad service from which to retrieve the model definitions
    """
    mdefs = pms.get_mdefs()
    
    print "Available model definitions:"
    print "----------------------------"
    for mdef in mdefs:
        print mdef


def ls_models(pms):
    """
    lists all the currently created models from the given service
    Arguments:
    pms -- the pymad service from which to retrieve the model definitions
    """
    
    print "Model instances:"
    print "----------------"
    for model in pms.get_models():
        print model.getName()
        
