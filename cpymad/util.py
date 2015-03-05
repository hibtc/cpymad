"""
Utility functions used in other parts of the pymad package.
"""
import collections
import re

from .types import Range, Constraint


__all__ = [
    'mad_quote',
    'is_identifier',
    'strip_element_suffix',
    'add_element_suffix',
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


# precompile regexes for performance:
_re_is_identifier = re.compile(r'^[a-z_]\w*$', re.IGNORECASE)
_re_element_suffix = re.compile(':\d+$')


def is_identifier(name):
    """Check if ``name`` is a valid identifier in MAD-X."""
    return bool(_re_is_identifier.match(name))


def strip_element_suffix(element_name):
    """
    Strip the :d suffix from an element name.

    The :d suffix is needed for some parts of the MAD-X API, but must not be
    used in other parts.
    """
    return _re_element_suffix.sub('', element_name)


def add_element_suffix(element_name):
    """
    Add a :1 suffix to an element name if missing.

    The :d suffix is needed for some parts of the MAD-X API, but must not be
    used in other parts.
    """
    if _re_element_suffix.search(element_name):
        return element_name
    return element_name + ':1'


def normalize_range_name(name):
    """Make element name usable as argument to the RANGE attribute."""
    if isinstance(name, tuple):
        return tuple(map(normalize_range_name, name))
    # MAD-X does not allow the ":d" suffix in the 'range' parameter string.
    # This means that name becomes less unique, but that's the only way right
    # now:
    name = strip_element_suffix(name)
    name = name.lower()
    if name.endswith('$end'):
        return '#e'
    if name.endswith('$start'):
        return '#s'
    return name


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
        begin, end = normalize_range_name((value.first, value.last))
        return key + '=' + begin + '/' + end
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
            return key + '=' + normalize_range_name(value)
        elif isinstance(value, collections.Mapping):
            begin, end = value['first'], value['last']
        else:
            begin, end = value[0], value[1]
        begin, end = normalize_range_name((str(begin), str(end)))
        return key + '=' + begin + '/' + end
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
