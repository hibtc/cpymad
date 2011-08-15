# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:58:50 2010

@author: kaifox
"""

class Enums:
    """
    collects shortcuts for variables that can be used in
    pymad
    """
    def __init__(self, jgw):
        """
        connects to the jmad service and initializes some convenient variables
        """
        self.JMadPlane = jgw.jvm.cern.accsoft.steering.jmad.domain.types.enums.JMadPlane
        self.MadxTwissVariable = jgw.jvm.cern.accsoft.steering.jmad.domain.var.enums.MadxTwissVariable
        self.MadxVarType = jgw.jvm.cern.accsoft.steering.jmad.util.MadxVarType
        self.KnobType = jgw.jvm.cern.accsoft.steering.jmad.domain.knob.KnobType