"""
Utility functions used in other parts of the pymad package.
"""
import re
import os
import sys
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from enum import Enum
from numbers import Number
import collections.abc as abc

import numpy as np

from cpymad.parsing import Parser
from cpymad.types import (
    Range, Constraint,
    PARAM_TYPE_LOGICAL, PARAM_TYPE_INTEGER,
    PARAM_TYPE_DOUBLE, PARAM_TYPE_STRING, PARAM_TYPE_CONSTRAINT,
    PARAM_TYPE_LOGICAL_ARRAY, PARAM_TYPE_INTEGER_ARRAY,
    PARAM_TYPE_DOUBLE_ARRAY, PARAM_TYPE_STRING_ARRAY)


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
    'ChangeDirectory',
]


# In CPython 3.6 dicts preserve insertion order (until deleting an element)
# Although, this is considered an implementation detail that should not be
# relied upon, we do so anyway:
ordered_keys = dict.keys if sys.version_info >= (3, 6) else sorted


def mad_quote(value: str) -> str:
    """Add quotes to a string value."""
    if '"' not in value:
        return '"' + value + '"'
    if "'" not in value:
        return "'" + value + "'"
    # MAD-X doesn't do any unescaping (otherwise I'd simply use `json.dumps`):
    raise ValueError("MAD-X unable to parse string with escaped quotes: {!r}"
                     .format(value))


def _fix_name(name: str) -> str:
    if name.startswith('_'):
        raise AttributeError("Unknown item: {!r}! Did you mean {!r}?"
                             .format(name, name.strip('_') + '_'))
    if name.endswith('_'):
        name = name[:-1]
    return name


# precompile regexes for performance:
def re_compile(s):
    return re.compile(s, re.IGNORECASE)


_re_is_identifier = re_compile(r'^[a-z_][a-z0-9_]*$')
_re_symbol = re_compile(r'([a-z_][a-z0-9._]*(->[a-z_][a-z0-9._]*(\[[0-9]+\])?)?)')
_re_element_internal = re_compile(r'^([a-z_][a-z0-9_.$]*)(:\d+)?$')
_re_element_external = re_compile(r'^([a-z_][a-z0-9_.$]*)(\[\d+\])?$')


def is_identifier(name: str) -> bool:
    """Check if ``name`` is a valid identifier in MAD-X."""
    return bool(_re_is_identifier.match(name))


def expr_symbols(expr: str) -> set:
    """
    Return all symbols names used in an expression.

    For now this includes not only variables but also element attributes (e.g.
    ``quad->k1``) as well as function names (e.g. ``sin``).
    """
    return {m[0] for m in _re_symbol.findall(expr)}


def name_from_internal(element_name: str) -> str:
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
                         .format(element_name)) from None
    if count is None or count == ':1':
        return name
    return name + '[' + count[1:] + ']'


def _parse_element_name(element_name: str) -> tuple:
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
                         .format(element_name)) from None
    if count is None:
        return name, None
    return name, int(count[1:-1])


def name_to_internal(element_name: str) -> str:
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


def normalize_range_name(name: str, elems=None) -> str:
    """Make element name usable as argument to the RANGE attribute."""
    if isinstance(name, tuple):
        return tuple(map(normalize_range_name, name))
    if '/' in name:
        return '/'.join(map(normalize_range_name, name.split('/')))
    name = name.lower()
    if name.endswith('$end') or name.endswith('$start'):
        if elems is None:
            return u'#s' if name.endswith('$start') else u'#e'
        else:
            return u'#s' if elems.index(name) == 0 else u'#e'
    return name


QUOTED_PARAMS = {'file', 'halofile', 'sectorfile', 'trueprofile'
                 'pipefile', 'trackfile', 'summary_file', 'filename',
                 'echo', 'title', 'text', 'format', 'dir'}


def format_param(key: str, value) -> str:
    """
    Format a single MAD-X command parameter.

    This is the old version that does not use type information from MAD-X. It
    is therefore not limited to existing MAD-X commands and attributes, but
    also less reliable for producing valid MAD-X statements.
    """
    if value is None:
        return None
    key = _fix_name(str(key).lower())
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
    # string is also a Sequence:
    elif isinstance(value, str):
        if key in QUOTED_PARAMS:
            return key + '=' + mad_quote(value)
        else:
            # MAD-X parses strings incorrectly, if followed by a boolean.
            # E.g.: "beam, sequence=s1, -radiate;" does NOT work! Therefore,
            # these values need to be quoted. (NOTE: MAD-X uses lower-case
            # internally and the quotes prevent automatic case conversion)
            return key + '=' + mad_quote(value.lower())
    # don't quote expressions:
    elif isinstance(value, str):
        return key + ':=' + value
    elif isinstance(value, abc.Sequence):
        return key + '={' + ','.join(map(str, value)) + '}'
    else:
        return key + '=' + str(value)


def _format_range(value) -> str:
    if isinstance(value, str):
        return normalize_range_name(value)
    elif isinstance(value, Range):
        begin, end = value.first, value.last
    else:
        begin, end = value
    begin, end = normalize_range_name((str(begin), str(end)))
    return begin + '/' + end


def format_cmdpar(cmd, key: str, value) -> str:
    """
    Format a single MAD-X command parameter.

    :param cmd: A MAD-X Command instance for which an argument is to be formatted
    :param key: name of the parameter
    :param value: argument value
    """
    key = _fix_name(str(key).lower())
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
        if isinstance(value, str):          return key + ':=' + value
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
        if isinstance(value, str):
            return key + '=' + format_str(value)
    if dtype == PARAM_TYPE_STRING_ARRAY:
        # NOTE: allowing single scalar value to STRING_ARRAY, mainly useful
        # for `match`, see above.
        if key == 'range' or isinstance(value, Range):
            if isinstance(value, list):
                return key + '={' + ','.join(map(_format_range, value)) + '}'
            return key + '=' + _format_range(value)
        if isinstance(value, str):
            return key + '=' + format_str(value)
        if isinstance(value, abc.Sequence):
            return key + '={' + ','.join(map(format_str, value)) + '}'

    raise TypeError('Unexpected command argument type: {}={!r} ({})'
                    .format(key, value, type(value)))


def format_command(*args, **kwargs) -> str:
    """
    Create a MAD-X command from its name and parameter list.

    :param cmd: base command (serves as template for parameter types)
    :param args: initial bareword command arguments (including command name!)
    :param kwargs: following named command arguments
    :returns: command string

    Examples:

    >>> format_command('twiss', sequence='lhc')
    'twiss, sequence=lhc;'

    >>> format_command('option', echo=True)
    'option, echo;'

    >>> format_command('constraint', betx=Constraint(max=3.13))
    'constraint, betx<3.13;'
    """
    cmd, args = args[0], args[1:]
    if isinstance(cmd, str):
        _args = [cmd] + list(args)
        _keys = ordered_keys(kwargs)
        _args += [format_param(k, kwargs[k]) for k in _keys]
    else:
        _args = [cmd.name] + list(args)
        _keys = ordered_keys(kwargs)
        _args += [format_cmdpar(cmd, k, kwargs[k]) for k in _keys]
    return u', '.join(filter(None, _args)) + ';'


# validation of MAD-X expressions

class T(Enum):
    """Terminal/token type."""
    WHITESPACE = 0
    LPAREN     = 1
    RPAREN     = 2
    COMMA      = 3
    SIGN       = 4
    OPERATOR   = 5
    SYMBOL     = 6
    NUMBER     = 7
    END        = 8

    __str__ = __repr__ = lambda self: self.name


class N(Enum):
    """Nonterminal symbol."""
    start            = 0
    expression       = 1
    expression_inner = 2
    expression_tail  = 3
    symbol_tail      = 4
    argument_list    = 5
    argument_tail    = 6

    __str__ = __repr__ = lambda self: self.name


grammar = {
    N.start: [
        [N.expression, T.END],
    ],
    N.expression: [
        [T.WHITESPACE, N.expression],
        [N.expression_inner],
    ],
    N.expression_inner: [
        [T.SIGN, N.expression],
        [T.LPAREN, N.expression, T.RPAREN, N.expression_tail],
        [T.NUMBER, N.expression_tail],
        [T.SYMBOL, N.symbol_tail],
    ],
    N.expression_tail: [
        [T.WHITESPACE, N.expression_tail],
        [T.SIGN, N.expression],
        [T.OPERATOR, N.expression],
        [],
    ],
    N.symbol_tail: [
        [T.WHITESPACE, N.symbol_tail],
        [T.LPAREN, N.argument_list, T.RPAREN, N.expression_tail],
        [T.SIGN, N.expression],
        [T.OPERATOR, N.expression],
        [],
    ],
    N.argument_list: [
        [T.WHITESPACE, N.argument_list],
        [N.expression_inner, N.argument_tail],
        [],
    ],
    N.argument_tail: [
        [T.COMMA, N.argument_list],
        [],
    ],
}


def _regex(expr: str) -> callable:
    regex = re.compile(expr)
    def match(text, i):
        m = regex.match(text[i:])
        return m.end() if m else 0
    return match


def _choice(choices: str) -> callable:
    def match(text, i):
        return 1 if text[i] in choices else 0
    return match


_expr_tokens = {
    T.WHITESPACE:   _choice(' \t'),
    T.LPAREN:       _choice('('),
    T.RPAREN:       _choice(')'),
    T.COMMA:        _choice(','),
    T.SIGN:         _choice('+-'),
    T.OPERATOR:     _choice('/*^'),
    T.SYMBOL:       _regex(r'[a-zA-Z_][a-zA-Z0-9_.]*(->[a-zA-Z_][a-zA-Z0-9_]*)?'),
    T.NUMBER:       _regex(r'(\d+(\.\d*)?|\.\d+)([eE][+\-]?\d+)?'),
}

_expr_parser = Parser(T, grammar, N.start)


class Token(namedtuple('Token', ['type', 'start', 'length', 'expr'])):

    @property
    def text(self):
        return self.expr[self.start:self.start + self.length]

    def __repr__(self):
        return '{}({!r})'.format(self.type, self.text)


def tokenize(tokens, expr: str):
    i = 0
    stop = len(expr)
    while i < stop:
        for toktype, tokmatch in tokens:
            length = tokmatch(expr, i)
            if length > 0:
                yield Token(toktype, i, length, expr)
                i += length
                break
        else:
            raise ValueError("Unknown token {!r} at {!r}"
                             .format(expr[i], expr[:i+1]))


def check_expression(expr: str):
    """
    Check if the given expression is a valid MAD-X expression that is safe to
    pass to :meth:`cpymad.madx.Madx.eval`.

    :param expr:
    :returns: True
    :raises ValueError: if the expression is ill-formed

    Note that this function only recognizes a sane subset of the expressions
    accepted by MAD-X and rejects valid but strange ones such as a number
    formatting '.' representing zero.
    """
    expr = expr.strip().lower()
    tokens = list(tokenize(list(_expr_tokens.items()), expr))
    tokens.append(Token(T.END, len(expr), 0, expr))
    _expr_parser.parse(tokens)  # raises ValueError
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


class ChangeDirectory:

    """
    Context manager for temporarily changing current working directory.

    :param str path: new path name
    :param _os: module with ``getcwd`` and ``chdir`` functions
    """

    # Note that the code is generic enough to be applied for any temporary
    # value patch, but we currently only need change directory, and therefore
    # have named it so.

    def __init__(self, path, chdir, getcwd):
        self._chdir = chdir
        self._getcwd = getcwd
        # Contrary to common implementations of a similar context manager,
        # we change the path immediately in the constructor. That enables
        # this utility to be used without any 'with' statement:
        if path:
            self._restore = getcwd()
            chdir(path)
        else:
            self._restore = None

    def __enter__(self):
        """Enter 'with' context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit 'with' context and restore old path."""
        if self._restore:
            self._chdir(self._restore)


@np.vectorize
def remove_count_suffix_from_name(name):
    """Return the :N suffix from an element name."""
    return name.rsplit(':', 1)[0]
