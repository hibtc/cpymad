##
# This file contains tool functions for madx.pyx
#



def _fixcmd(cmd):
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