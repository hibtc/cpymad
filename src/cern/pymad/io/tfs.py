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
'''
.. module:: tfs

Function to read tfs tables into Python objects

.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>
'''
import numpy
import os
from cern.pymad.domain import TfsTable, TfsSummary
    
def tfs(inputfile):
    '''
     Returns table and summary information
     as LookUp dictionaries. These extend on normal
     dictionary syntax. We recommend using this function
     for reading tfs files.
    '''
    table,params=tfsDict(inputfile)
    return TfsTable(table), TfsSummary(params)

def tfsDict(inputfile):
    '''
    .. py:function:: tfsDict(inputfile)

    Read a tfs table and returns table/summary info
    
    The function takes in a tfs file. It will add
    all parameters into one dictionary, and the table
    into another dictionary.

    :param string inputfile: tfs file, full path
    :raises ValueError: In case file path is not found
    :rtype: tuple containing dictionaries (tfs table , summary)

    See also: :mod:`pymad.domain.tfs`
    '''
    params={}
    if not os.path.isfile(inputfile):
        if os.path.isfile(inputfile+'.tfs'):
            inputfile+='.tfs'
        elif os.path.isfile(inputfile+'.TFS'):
            inputfile+='.TFS'
        else:
            raise ValueError("ERROR: "+inputfile+" is not a valid file path")
    f=file(inputfile,'r')
    l=f.readline()
    while(l):
        if l.strip()[0]=='@':
            _addParameter(params,l)
        if l.strip()[0]=='*': # beginning of vector list...
            names=l.split()[1:]
            table=_read_table(f,names)
        l=f.readline()
    return table, params

##
# Add parameter to object
# 
# Any line starting with an @ is a parameter.
# If that is found, this function should be called and given the line
# 
# @param line The line from the file that should be added
def _addParameter(params,line):
    lname=line.split()[1].lower()
    if line.split()[2]=='%le':
        params[lname]=float(line.split()[3])
    if line.split()[2][-1]=='s':
        params[lname]=line.split('"')[1]
    if line.split()[2]=='%d':
        params[lname]=int(line.split()[3])

##
# Reads in a table in tfs format.
# Input the file stream at the location
# where the names of the columns have just been read.
def _read_table(fstream,names):
    l=fstream.readline()
    types=[]
    table={}
    for n in names:
        table[n.lower()]=[]
    while(l):
        if l.strip()[0]=='$':
            types=l.split()[1:]
        else:
            for n,el in zip(names,l.split()):
                table[n.lower()].append(el)
        l=fstream.readline()
    for n,typ in zip(names,types):
        if typ=='%le':
            table[n.lower()]=numpy.array(table[n.lower()],dtype=float)
        elif typ=='%d':
            table[n.lower()]=numpy.array(table[n.lower()],dtype=int)
        elif typ=='%s':
            for k in xrange(len(table[n.lower()])):
                table[n.lower()][k]=table[n.lower()][k].split('"')[1]
    return table
