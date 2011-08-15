# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:20:15 2010

@author: kaifox
"""
from conversions import tofl
from conversions import tostr

def get_values(optic, madxvarname):
    """
    extract the values for the given madx-variable from the optcs object
    
    PARAMETERS:
    ===========
    optic: the object from which to extract the values
    madxvarname: the name of the madx-variable for which to extract the values
    """
    madxvar = pms.enums.MadxTwissVariable.fromMadxName(madxvarname) #@UndefinedVariable
    values = optic.getAllValues(madxvar)
    return tofl(values)

def get_names(optic):
    '''
    extracts the element names from the optics
    '''
    return tostr(optic.getNames())
    
