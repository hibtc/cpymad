# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:21:25 2010

@author: kaifox
"""
from run import get_pms 


def ls_mdefs(pms=None):
    """
    lists all the currently available model definitions from the given service
    
    Arguments:
    :param pms: the pymad service from which to retrieve the model definitions, 
    if this is None then the singleton is used
    """
    if pms is None:
        pms = get_pms()
        
    mdefs = pms.get_mdefs()
    
    print "Available model definitions:"
    print "----------------------------"
    for mdef in mdefs:
        print mdef


def ls_models(pms=None):
    """
    lists all the currently created models from the given service
    Arguments:
    :param pms: the pymad service from which to retrieve the model definitions, if this is None then the singleton is used
    """
    if pms is None:
        pms = get_pms()
        
    print "Model instances:"
    print "----------------"
    for model in pms.models:
        print model
        
