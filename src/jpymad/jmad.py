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