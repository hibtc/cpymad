##
# This file contains tool functions for madx.pyx
#
import collections
import re


from cern.pymad.io import tfs,tfsDict
from cern.pymad.domain.tfs import LookupDict


def _checkCommand(cmd):
    ''' give the lowercase version of the command
    this function does some sanity checks...'''
    if "stop;" in cmd or "exit;" in cmd:
        print("WARNING: found quit in command: "+cmd+"\n")
        print("Please use madx.finish() or just exit python (CTRL+D)")
        print("Command ignored")
        return False
    if cmd.split(',')>0 and "plot" in cmd.split(',')[0]:
        print("WARNING: Plot functionality does not work through pymadx")
        print("Command ignored")
        return False
    # All checks passed..
    return True

def _fixcmd(cmd):
    '''
    Makes sure command is sane.
    '''
    if not isinstance(cmd, basestring):
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
        if isinstance(madrange, basestring):
            return 'range='+madrange+','
        elif isinstance(madrange, collections.Sequence):
            return 'range='+madrange[0]+'/'+madrange[1]+','
        elif isinstance(madrange, collections.Mapping):
            return 'range='+madrange['first']+'/'+madrange['last']+','
        else:
            raise TypeError("Wrong range type/format")
    return ''

def _add_offsets(offsets):
    if offsets:
        return 'offsetelem="'+offsets+'",'
    return ''

def _sorted_items(kwargs):
    """Return dictionary items in canonicalized order."""
    if isinstance(kwargs, getattr(collections, 'OrderedDict', ())):
        return kwargs.items()
    else:
        return sorted(kwargs.items(), key=lambda i: i[0])

def _mad_command(cmd, *args, **kwargs):
    """
    Create a MAD-X command from its name and parameter list.

    @param cmd [string] name of the MAD command
    @params *args [list] ordered arguments to the MAD command
    @params **kwargs [dict] unordered arguments to the MAD command

    Examples:

    >>> print(_mad_command('twiss', ('sequence', 'lhc'), 'centre', dx=2, betx=3, bety=8).rstrip())
    twiss, sequence=lhc, centre, betx=3, bety=8, dx=2;

    >>> print(_mad_command('option', echo=False).rstrip())
    option, -echo;

    >>> print(_mad_command('constraint', ('betx', '<', 3.13), 'bety < 3.5').rstrip())
    constraint, betx<3.13, bety < 3.5;

    >>> print(_mad_command('constraint', **{'betx<3.13':True}).rstrip())
    constraint, betx<3.13;

    Note that alphabetic order is enforced on kwargs, such that results are
    always reproducible.

    """
    mad = cmd
    fullargs = list(args) + _sorted_items(kwargs)
    for arg in fullargs:
        if isinstance(arg, tuple):
            if len(arg) == 3:
                key, op, value = arg
            elif len(arg) == 2:
                key, value = arg
                op = '='
            elif len(arg) == 1:
                key, value = arg[0], True
            else:
                raise ValueError("Accepts only 1-to-3-tuples.")
        else:
            key = arg
            value = True

        key = str(key)
        if key.lower() == 'range':
            # NOTE: we need to cut the trailing ',' from _add_range:
            mad += ', ' + _add_range(value)[:-1]
        elif isinstance(value, bool):
            mad += ', ' + ('' if value else '-') + key
        else:
            mad += ', ' + key + op + str(value)
    mad += ';\n'
    return mad

def _mad_command_unpack(*arglists, **kwargs):
    """Create a MAD-X command from lists of its components."""
    args = []
    for v in arglists:
        if isinstance(v, basestring) or isinstance(v, tuple):
            args.append(v)
        elif isinstance(v, collections.Mapping):
            args += _sorted_items(v)
        elif isinstance(v, collections.Sequence):
            args += v
        else:
            raise TypeError("_call accepts only lists or dicts")
    return _mad_command(*args, **kwargs)

def _read_knobfile(filename, retdict):
    """
    Read the knobfile output of ENDMATCH.

    The input file is in a format like:

        k0sl_h1ms4v  :=+1.00000000e-04+0.00000000e+00*knob;
        k1_h3qd22    :=+8.57142860e-01+2.82559231e-01*knob;

    The result is a tuple `(final, initial)`. Where both entries are
    dictionaries or LookupDict depending on the `retdict` parameter.

    """
    result = {}
    initial = {}
    r_name = r'\s*(\w*)\s*'
    r_number = r'\s*([+-]?(?:\d+(?:\.\d*)?|\d*\.\d+)(?:[eE][+\-]?\d+)?)\s*'
    regex = re.compile('^' + r_name + ':=' + r_number + '([+-])' + r_number + r'\*\s*knob\s*;\s*$')
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = regex.match(line)
                if match:
                    knob_name = match.group(1)
                    initial_value = float(match.group(2))
                    variation = float(match.group(3) + match.group(4))
                    result[knob_name] = initial_value + variation
                    initial[knob_name] = initial_value
    except IOError:
        pass
    if retdict:
        return result, initial
    else:
        return LookupDict(result), LookupDict(initial)


