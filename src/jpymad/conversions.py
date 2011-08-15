# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:15:14 2010

@author: kaifox
"""

def tofl(values):
    """
    converts a general list to a list of floats
    """
    return [float(value) for value in values]

def tostr(values):
    """
    converts a general list to a list of strings
    """
    return [str(s) for s in values]