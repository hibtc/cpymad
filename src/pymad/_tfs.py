# -*- coding: utf-8 -*-
import numpy
import os
    
##
# @brief Read a tfs table and returns table/parameters
# 
# The function takes in a tfs file. It will add
# all parameters into one dictionary, and the table
# into another dictionary.
# Returns both.
def tfs(inputfile):
    params={}
    if type(inputfile)!=str:
        print("ERROR: inputfile must be a string")
        return None
    if not os.path.isfile(inputfile):
        if os.path.isfile(inputfile+'.tfs'):
            inputfile+='.tfs'
        elif os.path.isfile(inputfile+'.TFS'):
            inputfile+='.TFS'
        else:
            print("ERROR: "+inputfile+" is not a valid file path")
            return None
    f=file(inputfile,'r')
    l=f.readline()
    while(l):
        if l.strip()[0]=='@':
            _addParameter(params,l)
        if l.strip()[0]=='*': # beginning of vector list...
            names=l.split()[1:]
            table=_readTable(f,names)
        l=f.readline()
    return table,params

##
# Add parameter to object
# 
# Any line starting with an @ is a parameter.
# If that is found, this function should be called and given the line
# 
# @param line The line from the file that should be added
def _addParameter(params,line):
    if line.split()[2]=='%le':
        params[line.split()[1]]=float(line.split()[3])
    if line.split()[2][-1]=='s':
        params[line.split()[1]]=line.split('"')[1]
    if line.split()[2]=='%d':
        params[line.split()[1]]=int(line.split()[3])

##
# Reads in a table in tfs format.
# Input the file stream at the location
# where the names of the columns have just been read.
def _readTable(fstream,names):
    l=fstream.readline()
    types=[]
    table={}
    for n in names:
        table[n]=[]
    while(l):
        if l.strip()[0]=='$':
            types=l.split()[1:]
        else:
            for n,el in zip(names,l.split()):
                table[n].append(el)
        l=fstream.readline()
    for n,typ in zip(names,types):
        if typ=='%le':
            table[n]=numpy.array(table[n],dtype=float)
        elif typ=='%d':
            table[n]=numpy.array(table[n],dtype=int)
        elif typ=='%s':
            for k in xrange(len(table[n])):
                table[n][k]=table[n][k].split('"')[1]
    return table
