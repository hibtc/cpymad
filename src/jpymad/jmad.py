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
Created on Tue Nov 16 16:26:03 2010

@author: kaifox
"""


# os: needed for system calls
import os

# the url, where to find the jmad gui for download
JMAD_GUI_URL = r'http://abwww.cern.ch/ap/dist/accsoft/steering/accsoft-steering-jmad-gui/PRO/accsoft-steering-jmad-gui.jnlp'

def _start_jmad_gui():
    """
    starts the jmad gui from webstart.
    """
    cmd = 'javaws "' + JMAD_GUI_URL + '"'
    os.system(cmd)

def start_jmad():
    """
    starts jmad to later be able to connect to it.
    """
    _start_jmad_gui()
