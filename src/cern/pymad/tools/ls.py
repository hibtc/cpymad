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
        
    mdefs = pms.mdefs
    
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
        
