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
from jpymad.variables import Enums
"""
Created on Tue Nov 16 16:26:03 2010

@author: kaifox
"""

import os
import sys
from globals import JPyMadGlobals
from subprocess import Popen
from py4j.java_gateway import JavaGateway
from time import sleep

_SLEEP_INTERVAL = 0.1

class ConnectionFailedError(Exception):
    pass;

def _get_extension():
    if sys.platform.startswith('win'):
        return '.bat'
    else:
        return '.sh'

def _get_jmad_home():
    """
    returns the home directory of jmad
    """
    varname = 'JMAD_HOME'
    if os.environ.has_key(varname):
        return os.environ[varname]
    else:
        print "WARN: environment variable '" + varname + "' not set! Either set it to the root of jmad (pymadservice or gui), or start the gui manually."
        return None 

def _wait_for_file(filename, timeout=10.0):
    '''
    waits until the service writes the file
    '''
    time = 0.0
    while True:
        if os.path.isfile(filename) == True:
            break
        sleep(_SLEEP_INTERVAL)
        time = time + _SLEEP_INTERVAL
        if time > timeout:
            return False
        
    _delete_waitfile(filename, True)
    return True
        
def _delete_waitfile(filename, ignorefail=True):
    '''
    deletes the file, which is used to determine, if madx is ready
    '''
    if ignorefail:
        try:
            os.remove(filename)
        except:
            pass
    else:
        while True:
            if not os.path.exists(filename):
                break
            try:
                os.remove(filename)
                break # exit the loop if succesful
            except:
                print "WARN: failed to delete file'" + filename + "' try again in " + str(_SLEEP_INTERVAL) + " sec."
                sleep(_SLEEP_INTERVAL)


def _start(scriptname, jmadhome=None):
    """
    starts a jmad script
    """
    if jmadhome is None:
        jmadhome = _get_jmad_home()
        
    if jmadhome is None:
        return False
    
    waitfile = os.path.join(jmadhome, 'pymad-service-ready.out')
    _delete_waitfile(waitfile, False)
    
    cmd = os.path.join(jmadhome, scriptname + _get_extension())
    
    if not os.path.isfile(cmd):
        print "WARN: start script '" + cmd + "' does not exist. Cannot start."
        return False
    
    Popen(cmd, cwd=jmadhome)
    
    if _wait_for_file(waitfile):
        print "... started."
        return True
    else:
        print "Starting timed out ..."
        return False

def start_gui(jmadhome=None):
    """
    starts jmad to later be able to connect to it.
    """
    return _start('start-gui', jmadhome)

def start_pymadservice(jmadhome=None):
    """
    starts the nogui-version of the service
    """
    _start('start-pymadservice', jmadhome)

def stop():
    """
    ends the java process.
    """
    if not is_connected():
        print "Not connected to java, cannot stop the process! Try using 'connect()' first and then 'stop()'."
        return
    
    java_pymadservice = JPyMadGlobals.java_gateway.entry_point #@UndefinedVariable
    try:
        java_pymadservice.end()
        print "Something might have gone wrong, since we should get a disconnect exception, since the java process is ended"
    except:
        JPyMadGlobals.java_gateway = None
        JPyMadGlobals.jmad_service = None
        JPyMadGlobals.enums = None
        print "Stopping java process seemed to work."
    
    
def is_connected():
    """
    returns true, if a connection is established, false if not.
    """
    return not (JPyMadGlobals.jmad_service == None)
    
def connect():
    """
    Creates the gateway and the jmad service
    """
    
    # if the java-gateway was already initialized, then we have nothing to do!
    if is_connected():
        return
    
    if JPyMadGlobals.java_gateway == None:
        JPyMadGlobals.java_gateway = JavaGateway()
    
    if JPyMadGlobals.jmad_service == None:
        # the entry-point is directly the jmad service
        # test the connection
        try:
            str = JPyMadGlobals.java_gateway.jvm.java.lang.String #@UndefinedVariable
        except:
            raise ConnectionFailedError('Could not connect to Java-PyMadService. Maybe neither the jmad-service nor the jmad-gui is running?')
        else:
            # only assign the jmad_service, if no error occured
            JPyMadGlobals.jmad_service = JPyMadGlobals.java_gateway.entry_point.getJmadService() #@UndefinedVariable
    
            # now, that everything is connected, we can init the variables
            JPyMadGlobals.enums = Enums(JPyMadGlobals.java_gateway)
    
    return JPyMadGlobals.jmad_service
