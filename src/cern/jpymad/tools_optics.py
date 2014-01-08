#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:20:15 2010

@author: kaifox
"""
from __future__ import absolute_import

from .conversions import tofl
from .conversions import tostr

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

