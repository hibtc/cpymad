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
from __future__ import absolute_import

from .variables import Enums
"""
Created on Tue Nov 16 16:26:03 2010

@author: kaifox
"""

import os
import sys
from .globals import JPyMadGlobals
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
    return _get_env_var('JMAD_HOME')

def _wait_for_file(filename, timeout=10.0):
    '''
    waits until the service writes the file
    '''
    time = 0.0
    while True:
        if os.path.isfile(filename):
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
                print("WARN: failed to delete file'" + filename + "' try again in " + str(_SLEEP_INTERVAL) + " sec.")
                sleep(_SLEEP_INTERVAL)

def _get_env_var(varname):
    return os.environ.get(varname)

def _search_path_for_bin(binname):
    for d in _get_env_var("PATH").split(':'):
        if os.path.isfile(os.path.join(d,binname)):
            return d

def _start(scriptname, jmadhome=None):
    """
    starts a jmad script
    """
    if jmadhome is None:
        jmadhome = _get_jmad_home()

    if jmadhome is None: # YIL suggestion: also check system path..
        jmadhome = _search_path_for_bin(scriptname+ _get_extension())

    if jmadhome is None:
        print("WARNING: Could not locate jmad script "+scriptname)
        return False

    #waitfile = os.path.join(jmadhome, 'pymad-service-ready.out')
    waitfile = os.path.join(os.getcwd(),'pymad-service-ready.out')
    _delete_waitfile(waitfile, False)

    cmd = os.path.join(jmadhome, scriptname + _get_extension())

    if not os.path.isfile(cmd):
        print("WARN: start script '" + cmd + "' does not exist. Cannot start.")
        return False



    Popen([cmd,waitfile], cwd=jmadhome)

    if _wait_for_file(waitfile):
        print("... started.")
        return True
    else:
        print("Starting timed out ...")
        return False

def start_gui(jmadhome=None):
    """
    starts jmad to later be able to connect to it.
    """
    return _start('start-jmad-gui', jmadhome)

def start_pymadservice(jmadhome=None):
    """
    starts the nogui-version of the service
    """
    return _start('start-pymadservice', jmadhome)

def stop():
    """
    ends the java process.
    """
    if not is_connected():
        print("Not connected to java, cannot stop the process! Try using 'connect()' first and then 'stop()'.")
        return

    java_pymadservice = JPyMadGlobals.java_gateway.entry_point #@UndefinedVariable
    try:
        java_pymadservice.end()
        print("Something might have gone wrong, since we should get a disconnect exception, since the java process is ended")
    except:
        JPyMadGlobals.java_gateway = None
        JPyMadGlobals.jmad_service = None
        JPyMadGlobals.enums = None
        print("Stopping java process seemed to work.")


def is_connected():
    """
    returns true, if a connection is established, false if not.
    """
    return JPyMadGlobals.jmad_service is not None

def connect():
    """
    Creates the gateway and the jmad service
    """

    # if the java-gateway was already initialized, then we have nothing to do!
    if is_connected():
        return

    if JPyMadGlobals.java_gateway is None:
        JPyMadGlobals.java_gateway = JavaGateway()

    if JPyMadGlobals.jmad_service is None:
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
