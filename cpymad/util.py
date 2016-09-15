"""
Utility functions used in other parts of the pymad package.
"""
import collections
import re
import os
import tempfile
from contextlib import contextmanager

from .types import Range, Constraint, Expression


__all__ = [
    'mad_quote',
    'is_identifier',
    'name_from_internal',
    'name_to_internal',
    'mad_parameter',
    'mad_command',
    'temp_filename',
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
_re_element_internal = re.compile('^([a-z_][\w.$]*)(:\d+)?$', re.IGNORECASE)
_re_element_external = re.compile('^([a-z_][\w.$]*)(\[\d+\])?$', re.IGNORECASE)


def is_identifier(name):
    """Check if ``name`` is a valid identifier in MAD-X."""
    return bool(_re_is_identifier.match(name))


def name_from_internal(element_name):
    """
    Convert element name from internal representation to user API. Example:

    >>> name_from_internal("foo:1")
    foo
    >>> name_from_internal("foo:2")
    foo[2]

    Element names are stored with a ":d" suffix by MAD-X internally (data in
    node/sequence structs), but users must use the syntax "elem[d]" to access
    the corresponding elements. This function is used to transform any string
    coming from the user before passing it to MAD-X.
    """
    try:
        name, count = _re_element_internal.match(element_name).groups()
    except AttributeError:
        raise ValueError("Not a valid MAD-X element name: {!r}"
                         .format(element_name))
    if count is None or count == ':1':
        return name
    return name + '[' + count[1:] + ']'


def _parse_element_name(element_name):
    """
    Parse element name from user API. Example:

    >>> _parse_element_name("foo")
    (foo, None)
    >>> _parse_element_name("foo[2]")
    (foo, 2)

    See :func:`name_from_internal' for further information.
    """
    try:
        name, count = _re_element_external.match(element_name).groups()
    except AttributeError:
        raise ValueError("Not a valid MAD-X element name: {!r}"
                         .format(element_name))
    if count is None:
        return name, None
    return name, int(count[1:-1])


def name_to_internal(element_name):
    """
    Convert element name from user API to internal representation. Example:

    >>> name_to_external("foo")
    foo:1
    >>> name_to_external("foo[2]")
    foo:2

    See :func:`name_from_internal' for further information.
    """
    name, count = _parse_element_name(element_name)
    return name + ':' + str(1 if count is None else count)


def normalize_range_name(name):
    """Make element name usable as argument to the RANGE attribute."""
    if isinstance(name, tuple):
        return tuple(map(normalize_range_name, name))
    name = name.lower()
    if name.endswith('$end'):
        return '#e'
    if name.endswith('$start'):
        return '#s'
    return name


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
            constr.append(key + '>' + str(value.min))
        if value.max is not None:
            constr.append(key + '<' + str(value.max))
        if constr:
            return ', '.join(constr)
        else:
            return key + '=' + str(value.value)
    elif isinstance(value, Expression):
        return key + ':=' + value.expr
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
    _keys = sorted(kwargs)
    _args += [mad_parameter(k, kwargs[k]) for k in _keys]
    return ', '.join(filter(None, _args)) + ';'


def is_match_param(v):
    return v.lower() in ['rmatrix', 'chrom', 'beta0', 'deltap',
            'betx','alfx','mux','x','px','dx','dpx',
            'bety','alfy','muy','y','py','dy','dpy' ]


@contextmanager
def temp_filename():
    """Get filename for use within 'with' block and delete file afterwards."""
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    yield filename
    try:
        os.remove(filename)
    except OSError:
        pass
