"""
Utility functions used in other parts of the pymad package.
"""
import collections
import re
import os
import sys
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
    'check_expression'
    'temp_filename',
]


# In CPython 3.6 dicts preserve insertion order (until deleting an element)
# Although, this is considered an implementation detail that should not be
# relied upon, we do so anyway:
ordered_keys = dict.keys if sys.version_info >= (3,6) else sorted


try:
    unicode
except NameError:   # python3
    basestring = unicode = str


def mad_quote(value):
    """Add quotes to a string value."""
    quoted = repr(value)
    return quoted[1:] if quoted[0] == 'u' else quoted


# precompile regexes for performance:
re_compile = lambda s: re.compile(unicode(s), re.IGNORECASE)
_re_is_identifier = re_compile(r'^[a-z_][a-z0-9_]*$')
_re_element_internal = re_compile(r'^([a-z_][a-z0-9_.$]*)(:\d+)?$')
_re_element_external = re_compile(r'^([a-z_][a-z0-9_.$]*)(\[\d+\])?$')


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

    See :func:`name_from_internal` for further information.
    """
    name, count = _parse_element_name(element_name)
    return name + ':' + str(1 if count is None else count)


def normalize_range_name(name):
    """Make element name usable as argument to the RANGE attribute."""
    if isinstance(name, tuple):
        return tuple(map(normalize_range_name, name))
    name = name.lower()
    if name.endswith('$end'):
        return u'#e'
    if name.endswith('$start'):
        return u'#s'
    return name


def mad_parameter(key, value):
    """
    Format a single MAD-X command parameter.
    """
    key = str(key).lower()
    # the empty string was used in earlier versions in place of None:
    if value is None or value == '':
        return u''
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
            return u', '.join(constr)
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
    _keys = ordered_keys(kwargs)
    _args += [mad_parameter(k, kwargs[k]) for k in _keys]
    return u', '.join(filter(None, _args)) + ';'


def is_match_param(v):
    return v.lower() in ['rmatrix', 'chrom', 'beta0', 'deltap',
            'betx','alfx','mux','x','px','dx','dpx',
            'bety','alfy','muy','y','py','dy','dpy' ]


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
    pass to :meth:`cpymad.madx.Madx.evaluate`.

    :param str expr:
    :returns: True
    :raises ValueError: if the expression is ill-formed

    Note that this function only recognizes a sane subset of the expressions
    accepted by MAD-X and rejects valid but strange ones such as a number
    formatting '.' representing zero.
    """

    try:
        # handle instance of type Expression:
        expr = expr.expr
    except AttributeError:
        pass
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
            continue

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
    yield filename
    try:
        os.remove(filename)
    except OSError:
        pass
