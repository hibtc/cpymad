"""
Utility functions used in other parts of the pymad package.
"""
import collections

from .types import Range, Constraint


__all__ = [
    'mad_quote',
    'mad_parameter',
    'mad_command',
]


try:
    unicode
except NameError:   # python3
    basestring = unicode = str


def mad_quote(value):
    """Add quotes to a string value."""
    quoted = repr(value)
    return quoted[1:] if quoted[0] == 'u' else quoted


SOON = 1
SAFE = 10
LATE = 100
LAST = 1000


def mad_param_order(key, value):
    """
    Determine parameter order.

    Sadly, the parameter order actually matters - because of deficiencies in
    the MAD-X language and parser. For example the following cases can be
    problematic:

    - booleans after strings: "beam, sequence=s1, -radiate;"
    - after string lists: "select, flag=twiss, column=x,y, -full;"
    """
    key = str(key).lower()
    # the empty string was used in earlier versions in place of None:
    if value is None or value == '':
        return SAFE
    if isinstance(value, Range):
        return LATE
    elif isinstance(value, Constraint):
        return SAFE
    elif isinstance(value, bool):
        return SOON
    elif key == 'range':
        return LATE
    # check for basestrings before collections.Sequence, because every
    # basestring is also a Sequence:
    elif isinstance(value, basestring):
        return SAFE
    elif isinstance(value, collections.Sequence):
        if key == 'column':
            return LAST
        elif value and all(isinstance(v, basestring) for v in value):
            return LAST
        else:
            return SAFE
    else:
        return LATE


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
        elif isinstance(value, collections.Mapping):
            return key + '=' + str(value['first']) + '/' + str(value['last'])
        else:
            return key + '=' + str(value[0]) + '/' + str(value[1])
    # check for basestrings before collections.Sequence, because every
    # basestring is also a Sequence:
    elif isinstance(value, basestring):
        if key == 'file':
            return key + '=' + mad_quote(value)
        else:
            # MAD-X parses strings incorrectly, if followed by a boolean.
            # E.g.: "beam, sequence=s1, -radiate;" does NOT work! Therefore,
            # these values need to be quoted. (NOTE: MAD-X uses lower-case
            # internally and the quotes prevent automatic case conversion)
            return key + '=' + mad_quote(value.lower())
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
    _items = sorted(kwargs.items(), key=lambda v: mad_param_order(*v))
    _args += [mad_parameter(k, v) for k,v in _items]
    return ', '.join(filter(None, _args)) + ';'


def is_match_param(v):
    return v.lower() in ['rmatrix', 'chrom', 'beta0', 'deltap',
            'betx','alfx','mux','x','px','dx','dpx',
            'bety','alfy','muy','y','py','dy','dpy' ]
