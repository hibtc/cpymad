# cython: embedsignature=True
"""
Low level cython binding to MAD-X.

CAUTION: this module maps the global architecture of the MAD-X library
closely. That means that all functions will operate on a shared global
state! Take this into account when importing this module.

Probably, you want to interact with MAD-X via the cpymad.madx module. It
provides higher level abstraction and can deal with multiple instances of
MAD-X. Furthermore, it enhances the security by boxing all MAD-X calls into
a subprocess.
"""

from os import chdir, getcwd

import numpy as np      # Import the Python-level symbols of numpy
cimport numpy as cnp    # Import the C-level symbols of numpy

from cern.cpymad.types import Constraint, Expression
cimport cern.cpymad.clibmadx as clib


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
cnp.import_array()


# Remember whether start() was called
_madx_started = False


# Python-level binding to libmadx:
__all__ = [
    'started',
    'start',
    'finish',
    'input',
    'sequence_exists',
    'get_twiss',
    'get_beam',
    'get_active_sequence',
    'get_sequences',
    'table_exists',
    'get_table_list',
    'get_table_summary',
    'get_table_column',
    'get_elements',
    'get_expanded_elements',
    'is_expanded',
    'evaluate',
    # these are imported from 'os' for convenience in madx.Madx and should
    # not really be considered part of the public interface:
    'chdir',
    'getcwd',
]


def started():
    """
    Check whether MAD-X has been initialized.

    :returns: whether :func:`start` was called without matching :func:`finish`
    :rtype: bool
    """
    return _madx_started


def start():
    """
    Initialize MAD-X.
    """
    clib.madx_start()
    global _madx_started
    _madx_started = True


def finish():
    """
    Cleanup MAD-X.
    """
    clib.madx_finish()
    global _madx_started
    _madx_started = False


def input(cmd):
    """
    Pass one input command to MAD-X.

    :param str cmd: command to be executed by the MAD-X interpretor
    """
    cdef bytes _cmd = _cstr(cmd)
    clib.stolower_nq(_cmd)
    clib.pro_input(_cmd)


def sequence_exists(sequence):
    """
    Check if the sequence exists.

    :param str sequence: sequence name
    :returns: True if the sequence exists
    :rtype: bool
    """
    try:
        _find_sequence(sequence)
        return True
    except ValueError:
        return False


def get_twiss(sequence_name):
    """
    Get the last calculated twiss table for the given sequence.

    :param str sequence_name: sequence name
    :returns: twiss table name
    :rtype: str
    :raises ValueError: if the sequence name is invalid
    :raises RuntimeError: if the twiss table is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if not seq.tw_valid:
        raise RuntimeError("TWISS table invalid.")
    return _str(seq.tw_table.name)


def get_beam(sequence_name):
    """
    Get the beam associated to the sequence.

    :param str sequence_name: sequence name
    :returns: beam properties as set with the BEAM command
    :rtype: dict
    :raises ValueError: if the sequence name is invalid
    :raises RuntimeError: if the sequence has no associated beam
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if seq.beam is NULL or not seq.beam.beam_def:
        raise RuntimeError("No beam attached to {}".format(sequence_name))
    return _parse_command(seq.beam)


def get_active_sequence():
    """
    Get the name of the active sequence.

    :returns: name of active sequence
    :rtype: str
    :raises RuntimeError: if no sequence is activated
    """
    if clib.current_sequ is NULL:
        raise RuntimeError("No active sequence!")
    return _str(clib.current_sequ.name)


def get_sequences():
    """
    Get a list of all sequences currently in memory.

    :returns: sequence names
    :rtype: list
    """
    cdef clib.sequence_list* seqs = clib.madextern_get_sequence_list()
    cdef int i
    return [_str(seqs.sequs[i].name)
            for i in xrange(seqs.curr)]


def table_exists(table):
    """
    Check if the table exists.

    :param str table: table name
    :returns: True if the table exists
    :rtype: bool
    """
    cdef bytes _table = _cstr(table)
    return bool(clib.table_exists(_table))


def get_table_list():
    """
    Return list of all table names.

    :returns: table names
    :rtype: list
    """
    return [_str(clib.table_register.names.names[i])
            for i in xrange(clib.table_register.names.curr)]


def get_table_summary(table):
    """
    Get table summary.

    :param str table: table name
    :returns: mapping of {column: value}
    :rtype: dict
    """
    cdef bytes _table = _cstr(table)
    cdef clib.char_p_array* header = clib.table_get_header(_table)
    cdef int i
    if header is NULL:
        raise ValueError("No summary for table: {!r}".format(table))
    return dict([_split_header_line(header.p[i])
                 for i in xrange(header.curr)])


def get_table_columns(table):
    """
    Get a list of all columns in the table.

    :param str table: table name
    :returns: column names
    :rtype: list
    :raises ValueError: if the table name is invalid
    """
    cdef bytes _table = _cstr(table)
    cdef int index = clib.name_list_pos(_table, clib.table_register.names)
    if index == -1:
        raise ValueError("Invalid table: {!r}".format(table))
    return _name_list(clib.table_register.tables[index].columns, [2,3])


def get_table_column(table, column):
    """
    Get data from the specified table.

    :param str table: table name
    :param str column: column name
    :returns: the data in the requested column
    :rtype: numpy.array
    :raises ValueError: if the column cannot be found in the table
    :raises RuntimeError: if the column has unknown type

    CAUTION: Numeric data is wrapped in numpy arrays but not copied. Make
    sure to copy all data before invoking any further MAD-X commands! This
    is done automatically for you if using libmadx in a remote service
    (pickle serialization effectively copies the data).
    """
    cdef char** char_tmp
    cdef cnp.npy_intp shape[1]
    cdef bytes _table = _cstr(table)
    cdef bytes _column = _cstr(column)
    cdef clib.column_info info = clib.table_get_column(_table, _column)
    dtype = <bytes> info.datatype
    # double:
    if dtype == b'd':
        shape[0] = <cnp.npy_intp> info.length
        return cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, info.data)
    # string:
    elif dtype == b'S':
        char_tmp = <char**> info.data
        return np.array([char_tmp[i] for i in xrange(info.length)])
    # invalid:
    elif dtype == b'V':
        raise ValueError("Column {!r} is not in table {!r}."
                         .format(column, table))
    # unknown:
    else:
        raise RuntimeError("Unknown datatype {!r} in column {!r}."
                           .format(_str(dtype), column))


def get_elements(sequence_name):
    """
    Return list of all elements in the original sequence.

    :param str sequence_name: sequence name
    :returns: all elements in the original sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid.
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return [_get_node(seq.nodes.nodes[i], seq.ref_flag)
            for i in xrange(seq.nodes.curr)]


def get_expanded_elements(sequence_name):
    """
    Return list of all elements in the expanded sequence.

    :param str sequence_name: sequence name
    :returns: all elements in the expanded sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid.
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return [_get_node(seq.all_nodes[i], seq.ref_flag)
            for i in xrange(seq.n_nodes)]


def is_expanded(sequence_name):
    """
    Check whether a sequence has already been expanded.

    :param str sequence_name: sequence name
    :returns: expanded state of the sequence
    :rtype: bool
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return seq.n_nodes > 0


def evaluate(cmd):
    """
    Evaluates an expression and returns the result as double.

    :param str cmd: symbolic expression to evaluate
    :returns: numeric value of the expression
    :rtype: float

    NOTE: This function uses global variables as temporaries - which is in
    general an *extremely* bad design choice. Even though MAD-X uses global
    variables internally anyway, we should probably change this at some
    time.
    """
    # TODO: not sure about the flags (the magic constants 0, 2)
    cdef bytes _cmd = _cstr(cmd.lower())
    clib.pre_split(_cmd, clib.c_dum, 0)
    clib.mysplit(clib.c_dum.c, clib.tmp_p_array)
    expr = clib.make_expression(clib.tmp_p_array.curr, clib.tmp_p_array.p)
    value = clib.expression_value(expr, 2)
    clib.delete_expression(expr)
    return value


# Helper functions:

# The following functions are `cdef functions`, i.e. they can only be
# called from Cython code. It is necessary to use `cdef functions` whenever
# we want to pass parameters or return values with a pure C type.

_expr_types = [bool, int, float]

cdef _expr(clib.expression* expr,
           double value,
           int typeid=clib.PARAM_TYPE_DOUBLE):
    """Return a parameter value with an appropriate type."""
    _type = _expr_types[typeid]
    _value = _type(value)
    if expr is NULL or expr.string is NULL:
        return _value
    _expr = _str(expr.string).strip()
    if not _expr:
        return _value
    try:
        return _type(float(_expr))
    except ValueError:
        pass
    return Expression(_expr, _type(evaluate(_expr)), _type)


cdef _get_param_value(clib.command_parameter* par):

    """
    Get the value of a command parameter.

    :returns: value of the parameter
    :rtype: depends on the parameter type
    :raises ValueError: if the parameter type is invalid
    """

    if par.type in (clib.PARAM_TYPE_LOGICAL,
                    clib.PARAM_TYPE_INTEGER,
                    clib.PARAM_TYPE_DOUBLE):
        return _expr(par.expr, par.double_value, par.type)

    if par.type == clib.PARAM_TYPE_STRING:
        return _str(par.string)

    if par.type == clib.PARAM_TYPE_CONSTRAINT:
        if par.c_type == clib.CONSTR_TYPE_MIN:
            return Constraint(min=_expr(par.min_expr, par.c_min))
        if par.c_type == clib.CONSTR_TYPE_MAX:
            return Constraint(max=_expr(par.max_expr, par.c_max))
        if par.c_type == clib.CONSTR_TYPE_BOTH:
            return Constraint(min=_expr(par.min_expr, par.c_min),
                              max=_expr(par.max_expr, par.c_max))
        if par.c_type == clib.CONSTR_TYPE_VALUE:
            return Constraint(val=_expr(par.expr, par.double_value))

    if par.type in (clib.PARAM_TYPE_INTEGER_ARRAY, clib.PARAM_TYPE_DOUBLE_ARRAY):
        return [
            _expr(NULL if par.expr_list is NULL else par.expr_list.list[i],
                  par.double_array.a[i],
                  par.type - clib.PARAM_TYPE_LOGICAL_ARRAY)
            for i in xrange(par.double_array.curr)
        ]

    if par.type == clib.PARAM_TYPE_STRING_ARRAY:
        return [_str(par.m_string.p[i])
                for i in xrange(par.m_string.curr)]

    raise ValueError("Unknown parameter type: {}".format(par.type))


cdef _parse_command(clib.command* cmd):
    """
    Get the values of all parameters of a command.

    :returns: the command parameters
    :rtype: dict
    """
    # generator expressions are not yet supported in cdef functions, so
    # let's do it the hard way:
    res = {}
    cdef int i
    for i in xrange(cmd.par.curr):
        name = _str(cmd.par.parameters[i].name)
        res[name] = _get_param_value(cmd.par.parameters[i])
    return res


# The 'except NULL' clause is needed to forward exceptions from cdef
# functions with C return values, see:
# http://docs.cython.org/src/userguide/language_basics.html#error-return-values
cdef clib.sequence* _find_sequence(sequence_name) except NULL:
    """
    Get pointer to the C sequence struct of the specified sequence or NULL.

    :param str sequence_name: sequence name
    :raises ValueError: if the sequence can not be found
    """
    cdef bytes _sequence_name = _cstr(sequence_name)
    cdef clib.sequence_list* seqs = clib.madextern_get_sequence_list()
    cdef int index = clib.name_list_pos(_sequence_name, seqs.list)
    if index == -1:
        raise ValueError("Invalid sequence: {}".format(sequence_name))
    return seqs.sequs[index]


cdef _split_header_line(header_line):
    """Parse a table header value."""
    _, key, kind, value = _str(header_line).split(None, 3)
    if kind == "%le":
        return key, float(value)    # convert to number
    elif kind.endswith('s'):
        return key, value[1:-1]     # strip quotes from string
    else:
        return key, value           #


cdef _name_list(clib.name_list* names, inform):
    """Return a python list of names for the name_list."""
    cdef int i
    return [names.names[i] for i in xrange(names.curr)
            if names.inform[i] in inform]


cdef _str(char* s):
    """Decode C string to python string."""
    if s is NULL:
        # Returning an empty string will make the type of the parameter
        # transparent to the inspecting code:
        return ""
    return s.decode('utf-8')


cdef bytes _cstr(s):
    """Encode python string to C string."""
    if s is None:
        return b""
    return <bytes> s.encode('utf-8')


cdef _get_node(clib.node* node, int ref_flag):
    """Return dictionary with node + element attributes."""
    if node.p_elem is NULL:
        # Maybe this is a valid case, but better detect it with boom!
        raise RuntimeError("Empty node or subsequence! Please report this incident!")
    # normalize 'at' value to node entry:
    cdef double at = node.at_value
    if ref_flag == clib.REF_CENTER:
        at -= node.length / 2
    elif ref_flag == clib.REF_EXIT:
        at -= node.length
    data = _get_element(node.p_elem)
    data.update({'name': _str(node.name),
                 'type': _str(node.base_name),
                 'at': at})
    return data


cdef _get_element(clib.element* elem):
    """Return dictionary with element attributes."""
    return _parse_command(elem.def_)
