##
# This file contains tool functions for madx.pyx
#
import collections
import re

#from .types import LookupDict

from .types import LookupDict, Range, Constraint


try:
    unicode
except NameError:   # python3
    basestring = unicode = str


def is_word(value):
    """Check if value is a MAD-X identifier."""
    return value.isalnum() and value[0].isalpha()


def mad_quote(value):
    """Add quotes to a string value."""
    if is_word(value):
        return value
    quoted = repr(value)
    return quoted[1:] if quoted[0] == 'u' else quoted


def mad_parameter(key, value):
    """
    Format a single MAD-X command parameter.
    """
    key = str(key).lower()
    # the empty string was used in earlier versions in place of None:
    if value is None or value == '':
        return ''
    if isinstance(value, Range):
        return key + '=' + value.first + '/' + value.last
    elif isinstance(value, Constraint):
        constr = []
        if value.min is not None:
            constr.append(key + '>' + value.min)
        if value.max is not None:
            constr.append(key + '<' + value.max)
        if constr:
            return ', '.join(constr)
        else:
            return key + '=' + value.value
    elif isinstance(value, bool):
        return ('' if value else '-') + key
    elif key == 'range':
        if isinstance(value, basestring):
            return key + '=' + value
        else:
            return key + '=' + str(value[0]) + '/' + str(value[1])
    elif isinstance(value, basestring):
        return key + '=' + mad_quote(value)
    elif isinstance(value, collections.Sequence):
        if key == 'column':
            return key + '=' + ','.join(value)
        elif value and all(isinstance(v, basestring) for v in value):
            return key + '=' + ','.join(value)
        else:
            return key + '={' + ','.join(map(str, value)) + '}'
    else:
        return key + '=' + str(value)


def mad_command(*args, **kwargs):
    """
    Create a MAD-X command from its name and parameter list.

    :param args: initial bareword command arguments (including command name!)
    :param kwargs: following named command arguments
    :returns: command string
    :rtype: str

    Examples:

    >>> mad_command('twiss', sequence='lhc')
    'twiss, sequence=lhc;'

    >>> mad_command('option', echo=True)
    'option, echo;'

    >>> mad_command('constraint', betx=Constraint(max=3.13))
    'constraint, betx<3.13;'
    """
    _args = list(args)
    _args += [mad_parameter(k, v) for k,v in kwargs.items()]
    return ', '.join(filter(None, _args)) + ';'


# legacy functions:

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


