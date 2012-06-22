##
# This file contains tool functions for madx.pyx
#

from pymad.io import tfs,tfsDict


def _fixcmd(cmd):
    '''
    Makes sure command is sane.
    '''
    if type(cmd)!=str and type(cmd)!=unicode:
        raise TypeError("ERROR: input must be a string, not "+str(type(cmd)))
    if len(cmd.strip())==0:
        return 0
    if cmd.strip()[-1]!=';':
        cmd+=';'
    # for very long commands (probably parsed in from a file)
    # we split and only run one line at the time.
    if len(cmd)>10000:
        cmd=cmd.split('\n')
    return cmd

def _get_dict(tmpfile,retdict):
    '''
     Returns a dictionary from the temporary file.
    '''
    if retdict:
        return tfsDict(tmpfile)
    return tfs(tmpfile)

def _add_range(madrange):
    if madrange:
        if type(madrange)==str:
            return 'range='+madrange+','
        elif type(madrange)==list:
            return 'range='+madrange[0]+'/'+madrange[1]+','
        elif type(madrange)==dict:
            return 'range='+madrange['first']+'/'+madrange['last']+','
        else:
            raise TypeError("Wrong range type/format")
    return ''

def _add_offsets(offsets):
    if offsets:
        return 'offsetelem="'+offsets+'",'
    return ''
