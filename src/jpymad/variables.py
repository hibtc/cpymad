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
    def __init__(self, java_gateway):
        """
        connects to the jmad service and initializes some convenient variables
        """
        self.JMadPlane = java_gateway.jvm.cern.accsoft.steering.jmad.domain.types.enums.JMadPlane
        self.MadxTwissVariable = java_gateway.jvm.cern.accsoft.steering.jmad.domain.var.enums.MadxTwissVariable
        self.MadxVarType = java_gateway.jvm.cern.accsoft.steering.jmad.util.MadxVarType
        self.KnobType = java_gateway.jvm.cern.accsoft.steering.jmad.domain.knob.KnobType