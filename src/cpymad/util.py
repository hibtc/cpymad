"""
Utility functions used in other parts of the pymad package.
"""
import re
import os
import sys
import tempfile
from contextlib import contextmanager
from numbers import Number
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from .types import Range, Constraint


__all__ = [
    'mad_quote',
    'is_identifier',
    'name_from_internal',
    'name_to_internal',
    'format_param',
    'format_cmdpar',
    'format_command',
    'check_expression',
    'temp_filename',
]


# In CPython 3.6 dicts preserve insertion order (until deleting an element)
# Although, this is considered an implementation detail that should not be
# relied upon, we do so anyway:
ordered_keys = dict.keys if sys.version_info >= (3, 6) else sorted


try:
    unicode
except NameError:   # python3
    basestring = unicode = str

NoneType = type(None)


def mad_quote(value):
    """Add quotes to a string value."""
    if '"' not in value:
        return '"' + value + '"'
    if "'" not in value:
        return "'" + value + "'"
    # MAD-X doesn't do any unescaping (otherwise I'd simply use `json.dumps`):
    raise ValueError("MAD-X unable to parse string with escaped quotes: {!r}"
                     .format(value))


# precompile regexes for performance:
re_compile = lambda s: re.compile(unicode(s), re.IGNORECASE)
_re_is_identifier = re_compile(r'^[a-z_][a-z0-9_]*$')
_re_symbol = re_compile(r'([a-z_][a-z0-9._]*(->[a-z_][a-z0-9._]*(\[[0-9]+\])?)?)')
_re_element_internal = re_compile(r'^([a-z_][a-z0-9_.$]*)(:\d+)?$')
_re_element_external = re_compile(r'^([a-z_][a-z0-9_.$]*)(\[\d+\])?$')


def is_identifier(name):
    """Check if ``name`` is a valid identifier in MAD-X."""
    return bool(_re_is_identifier.match(name))


def expr_symbols(expr):
    """
    Return all symbols names used in an expression.

    For now this includes not only variables but also element attributes (e.g.
    ``quad->k1``) as well as function names (e.g. ``sin``).
    """
    return {m[0] for m in _re_symbol.findall(expr)}


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

    See :func:`name_from_internal` for further information.
    """
    name, count = _parse_element_name(element_name)
    return name + ':' + str(1 if count is None else count)


def normalize_range_name(name):
    """Make element name usable as argument to the RANGE attribute."""
    if isinstance(name, tuple):
        return tuple(map(normalize_range_name, name))
    if '/' in name:
        return '/'.join(map(normalize_range_name, name.split('/')))
    name = name.lower()
    if name.endswith('$end'):
        return u'#e'
    if name.endswith('$start'):
        return u'#s'
    return name


PARAM_TYPE_LOGICAL       = 0
PARAM_TYPE_INTEGER       = 1
PARAM_TYPE_DOUBLE        = 2
PARAM_TYPE_STRING        = 3
PARAM_TYPE_CONSTRAINT    = 4
PARAM_TYPE_LOGICAL_ARRAY = 10
PARAM_TYPE_INTEGER_ARRAY = 11
PARAM_TYPE_DOUBLE_ARRAY  = 12
PARAM_TYPE_STRING_ARRAY  = 13

VAR_TYPE_CONST    = 0
VAR_TYPE_DIRECT   = 1
VAR_TYPE_DEFERRED = 2
VAR_TYPE_STRING   = 3

QUOTED_PARAMS = {'file', 'halofile', 'sectorfile', 'trueprofile'
                 'pipefile', 'trackfile', 'summary_file', 'filename',
                 'echo', 'title', 'text', 'format'}


def format_param(key, value):
    """
    Format a single MAD-X command parameter.

    This is the old version that does not use type information from MAD-X. It
    is therefore not limited to existing MAD-X commands and attributes, but
    also less reliable for producing valid MAD-X statements.
    """
    if value is None:
        return None
    key = str(key).lower()
    if isinstance(value, Constraint):
        constr = []
        if value.min is not None:
            constr.append(key + '>' + str(value.min))
        if value.max is not None:
            constr.append(key + '<' + str(value.max))
        if constr:
            return u', '.join(constr)
        else:
            return key + '=' + str(value.val)
    elif isinstance(value, bool):
        return key + '=' + str(value).lower()
    elif key == 'range' or isinstance(value, Range):
        return key + '=' + _format_range(value)
    # check for basestrings before abc.Sequence, because every
    # basestring is also a Sequence:
    elif isinstance(value, basestring):
        if key in QUOTED_PARAMS:
            return key + '=' + mad_quote(value)
        else:
            # MAD-X parses strings incorrectly, if followed by a boolean.
            # E.g.: "beam, sequence=s1, -radiate;" does NOT work! Therefore,
            # these values need to be quoted. (NOTE: MAD-X uses lower-case
            # internally and the quotes prevent automatic case conversion)
            return key + '=' + mad_quote(value.lower())
    # don't quote expressions:
    elif isinstance(value, basestring):
        return key + ':=' + value
    elif isinstance(value, abc.Sequence):
        return key + '={' + ','.join(map(str, value)) + '}'
    else:
        return key + '=' + str(value)


def _format_range(value):
    if isinstance(value, basestring):
        return normalize_range_name(value)
    elif isinstance(value, Range):
        begin, end = value.first, value.last
    else:
        begin, end = value
    begin, end = normalize_range_name((str(begin), str(end)))
    return begin + '/' + end


def format_cmdpar(cmd, key, value):
    """
    Format a single MAD-X command parameter.
    """
    cmdpar = cmd.cmdpar[key]
    dtype = cmdpar.dtype
    # the empty string was used in earlier versions in place of None:
    if value is None or value == '':
        return u''

    # NUMERIC
    if dtype == PARAM_TYPE_LOGICAL:
        if isinstance(value, bool):         return key + '=' + str(value).lower()
    if dtype in (PARAM_TYPE_LOGICAL,
                 PARAM_TYPE_INTEGER,
                 PARAM_TYPE_DOUBLE,
                 PARAM_TYPE_CONSTRAINT,
                 # NOTE: allow passing scalar values to numeric arrays, mainly
                 # useful because many of the `match` command parameters are
                 # arrays, but we usually call it with a single sequence and
                 # would like to treat it similar to the `twiss` command:
                 PARAM_TYPE_LOGICAL_ARRAY,
                 PARAM_TYPE_INTEGER_ARRAY,
                 PARAM_TYPE_DOUBLE_ARRAY):
        if isinstance(value, bool):         return key + '=' + str(int(value))
        if isinstance(value, Number):       return key + '=' + str(value)
        if isinstance(value, basestring):   return key + ':=' + value
    if dtype == PARAM_TYPE_CONSTRAINT:
        if isinstance(value, Constraint):
            constr = []
            if value.min is not None:
                constr.append(key + '>' + str(value.min))
            if value.max is not None:
                constr.append(key + '<' + str(value.max))
            if constr:
                return u', '.join(constr)
            else:
                return key + '=' + str(value.val)
    if dtype in (PARAM_TYPE_LOGICAL_ARRAY,
                 PARAM_TYPE_INTEGER_ARRAY,
                 PARAM_TYPE_DOUBLE_ARRAY):
        if isinstance(value, abc.Sequence):
            if all(isinstance(v, Number) for v in value):
                return key + '={' + ','.join(map(str, value)) + '}'
            else:
                return key + ':={' + ','.join(map(str, value)) + '}'

    # STRING
    def format_str(value):
        if key in QUOTED_PARAMS:
            return mad_quote(value)
        # NOTE: MAD-X stops parsing the current argument as soon as it
        # encounters another parameter name of the current command:
        elif is_identifier(value) and value not in cmd:
            return value
        else:
            return mad_quote(value.lower())
    if dtype == PARAM_TYPE_STRING:
        if key == 'range' or isinstance(value, Range):
            return key + '=' + _format_range(value)
        if isinstance(value, basestring):
            return key + '=' + format_str(value)
    if dtype == PARAM_TYPE_STRING_ARRAY:
        # NOTE: allowing single scalar value to STRING_ARRAY, mainly useful
        # for `match`, see above.
        if key == 'range' or isinstance(value, Range):
            if isinstance(value, list):
                return key + '={' + ','.join(map(_format_range, value)) + '}'
            return key + '=' + _format_range(value)
        if isinstance(value, basestring):
            return key + '=' + format_str(value)
        if isinstance(value, abc.Sequence):
            return key + '={' + ','.join(map(format_str, value)) + '}'

    raise TypeError('Unexpected command argument type: {}={!r} ({})'
                    .format(key, value, type(value)))


def format_command(*args, **kwargs):
    """
    Create a MAD-X command from its name and parameter list.

    :param cmd: base command (serves as template for parameter types)
    :param args: initial bareword command arguments (including command name!)
    :param kwargs: following named command arguments
    :returns: command string
    :rtype: str

    Examples:

    >>> format_command('twiss', sequence='lhc')
    'twiss, sequence=lhc;'

    >>> format_command('option', echo=True)
    'option, echo;'

    >>> format_command('constraint', betx=Constraint(max=3.13))
    'constraint, betx<3.13;'
    """
    cmd, args = args[0], args[1:]
    if isinstance(cmd, basestring):
        _args = [cmd] + list(args)
        _keys = ordered_keys(kwargs)
        _args += [format_param(k, kwargs[k]) for k in _keys]
    else:
        _args = [cmd.name] + list(args)
        _keys = ordered_keys(kwargs)
        _args += [format_cmdpar(cmd, k, kwargs[k]) for k in _keys]
    return u', '.join(filter(None, _args)) + ';'


# validation of MAD-X expressions

def _regex(expr):
    regex = re.compile(unicode(expr))
    def match(text, i):
        m = regex.match(text[i:])
        return m.end() if m else 0
    return match


def _choice(choices):
    def match(text, i):
        return 1 if text[i] in choices else 0
    return match


_expr_tokens = [
    ('WHITESPACE',  _choice(' \t')),
    ('LPAREN',      _choice('(')),
    ('RPAREN',      _choice(')')),
    ('OPERATOR',    _choice('+-/*^')),
    ('SYMBOL',      _regex(r'[a-zA-Z_][a-zA-Z0-9_.]*(->[a-zA-Z_][a-zA-Z0-9_]*)?')),
    ('NUMBER',      _regex(r'(\d+(\.\d*)?|\.\d+)([eE][+\-]?\d+)?')),
]


def _tokenize(tokens, expr):
    i = 0
    stop = len(expr)
    while i < stop:
        for tokname, tokmatch in tokens:
            l = tokmatch(expr, i)
            if l > 0:
                yield tokname, i, l
                i += l
                break
        else:
            raise ValueError("Unknown token {!r} at {!r}"
                             .format(expr[i], expr[:i+1]))


def check_expression(expr):

    """
    Check if the given expression is a valid MAD-X expression that is safe to
    pass to :meth:`cpymad.madx.Madx.eval`.

    :param str expr:
    :returns: True
    :raises ValueError: if the expression is ill-formed

    Note that this function only recognizes a sane subset of the expressions
    accepted by MAD-X and rejects valid but strange ones such as a number
    formatting '.' representing zero.
    """

    expr = expr.strip()

    def unexpected(tok, i, l):
        return "Unexpected {} {!r} after {!r} in expression: {!r}".format(
            tok, expr[i:i+l], expr[:i], expr)

    _EXPRESSION = 'expression'
    _OPERATOR = 'operator'
    _ATOM = 'atom'
    expect = _EXPRESSION
    paren_level = 0

    # expr =
    #       symbol | number
    #   |   '(' expr ')'
    #   |   expr op expr

    for tok, i, l in _tokenize(_expr_tokens, expr):
        # ignore whitespace
        if tok == 'WHITESPACE':
            pass
        # expr = symbol | number
        elif tok in ('SYMBOL', 'NUMBER'):
            if expect not in (_EXPRESSION, _ATOM):
                raise ValueError(unexpected(tok, i, l))
            expect = _OPERATOR
        # expr = '(' expr ')'
        elif tok == 'LPAREN':
            if expect not in (_EXPRESSION, _ATOM):
                raise ValueError(unexpected(tok, i, l))
            paren_level += 1
            expect = _EXPRESSION
        elif tok == 'RPAREN':
            if expect != _OPERATOR:
                raise ValueError(unexpected(tok, i, l))
            paren_level -= 1
            expect = _OPERATOR
        # expr = expr op expr
        elif tok == 'OPERATOR':
            if expect == _OPERATOR:
                expect = _EXPRESSION
            elif expect == _EXPRESSION and expr[i] in '+-':
                expect = _ATOM
            else:
                raise ValueError(unexpected(tok, i, l))

    if expect != _OPERATOR:
        raise ValueError("Unexpected end-of-string in expression: {!r}"
                         .format(expr))
    if paren_level != 0:
        raise ValueError("{} unclosed left-parens in expression: {!r}"
                         .format(paren_level, expr))

    return True


# misc

@contextmanager
def temp_filename():
    """Get filename for use within 'with' block and delete file afterwards."""
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    try:
        yield filename
    finally:
        try:
            os.remove(filename)
        except OSError:
            pass


@contextmanager
def suppress(*exceptions):
    """Compat for contextlib.suppress for python < 3.4."""
    try:
        yield None
    except exceptions:
        pass
