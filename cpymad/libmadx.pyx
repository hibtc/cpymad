# cython: embedsignature=True
"""
Low level cython binding to MAD-X.

CAUTION: Do not import this module directly! Use :class:`Madx` instead.

- Importing this module means loading MAD-X directly into the process space.
  This means that any crash in the (extremely fragile) MAD-X interpreter will
  crash the importing process with it.
- All functions in this module operate on a shared global state.
- this module exposes a very C-ish API that is not convenient to work with.
"""

from os import chdir, getcwd

import ctypes
import numpy as np      # Import the Python-level symbols of numpy

# Import a large-enough integer type to hold pointer, see also:
# http://grokbase.com/t/gg/cython-users/134b21rga8/passing-callback-pointers-to-python-and-back
cdef extern from "pyport.h":
    ctypedef int Py_intptr_t

from cpymad.types import Constraint, Expression
cimport cpymad.clibmadx as clib


# Remember whether start() was called
_madx_started = False


# Python-level binding to libmadx:
__all__ = [
    # MAD-X version
    'get_version_number',
    'get_version_date',

    # control the interpreter
    'is_started',
    'start',
    'finish',
    'input',
    'evaluate',

    # iterate sequences
    'sequence_exists',
    'get_sequence_names',
    'get_sequence_count',
    'get_active_sequence_name',

    # sequence access
    'get_sequence_twiss_table_name',
    'get_sequence_beam',
    'is_sequence_expanded',

    # iterate tables
    'table_exists',
    'get_table_names',
    'get_table_count',

    # table access
    'get_table_summary',
    'get_table_column_names',
    'get_table_column',

    # sequence element list access
    'get_element',
    'get_element_index',
    'get_element_index_by_position',
    'get_element_count',

    # expanded sequence element access
    'get_expanded_element',
    'get_expanded_element_index',
    'get_expanded_element_index_by_position',
    'get_expanded_element_count',

    # these are imported from 'os' for convenience in madx.Madx and should
    # not really be considered part of the public interface:
    'chdir',
    'getcwd',
]


def _get_rightmost_word(sentence):
    """Get the work right of the rightmost space character."""
    return sentence.rsplit(' ', 1)[-1]


def get_version_number():
    """
    Get the version number of loaded MAD-X interpreter.

    :returns: full version number
    :rtype: str
    """
    return _get_rightmost_word(_str(clib.version_name))


def get_version_date():
    """
    Get the release date of loaded MAD-X interpreter.

    :returns: release date in YYYY.MM.DD format
    :rtype: str
    """
    return _get_rightmost_word(_str(clib.version_date))


def is_started():
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

    :param str cmd: command to be executed by the MAD-X interpreter
    """
    cdef bytes _cmd = _cstr(cmd)
    clib.stolower_nq(_cmd)
    clib.pro_input(_cmd)


def sequence_exists(sequence_name):
    """
    Check if the sequence exists.

    :param str sequence: sequence name
    :returns: True if the sequence exists
    :rtype: bool
    """
    try:
        _find_sequence(sequence_name)
        return True
    except ValueError:
        return False


def get_sequence_twiss_table_name(sequence_name):
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


def get_sequence_beam(sequence_name):
    """
    Get the beam associated to the sequence.

    :param str sequence_name: sequence name
    :returns: beam properties as set with the BEAM command (and some more)
    :rtype: dict
    :raises ValueError: if the sequence name is invalid
    :raises RuntimeError: if the sequence has no associated beam
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if seq.beam is NULL or not seq.beam.beam_def:
        raise RuntimeError("No beam attached to {}".format(sequence_name))
    return _parse_command(seq.beam)


def get_active_sequence_name():
    """
    Get the name of the active sequence.

    :returns: name of active sequence
    :rtype: str
    :raises RuntimeError: if no sequence is activated
    """
    if clib.current_sequ is NULL:
        raise RuntimeError("No active sequence!")
    return _str(clib.current_sequ.name)


def get_sequence_names():
    """
    Get a list of all sequences currently in memory.

    :returns: sequence names
    :rtype: list
    """
    cdef clib.sequence_list* seqs = clib.madextern_get_sequence_list()
    cdef int i
    return [_str(seqs.sequs[i].name)
            for i in xrange(seqs.curr)]


def get_sequence_count():
    """
    Get the number of all sequences currently in memory.

    :returns: number of sequences
    :rtype: int
    """
    return clib.madextern_get_sequence_list().curr


def table_exists(table_name):
    """
    Check if the table exists.

    :param str table: table name
    :returns: True if the table exists
    :rtype: bool
    """
    cdef bytes _table_name = _cstr(table_name)
    return bool(clib.table_exists(_table_name))


def get_table_names():
    """
    Return list of all table names.

    :returns: table names
    :rtype: list
    """
    return [_str(clib.table_register.names.names[i])
            for i in xrange(clib.table_register.names.curr)]


def get_table_count():
    """
    Return number of existing tables.

    :returns: number of tables in memory
    :rtype: int
    """
    return clib.table_register.names.curr


def get_table_summary(table_name):
    """
    Get table summary.

    :param str table_name: table name
    :returns: mapping of {column: value}
    :rtype: dict
    """
    cdef bytes _table_name = _cstr(table_name)
    cdef clib.char_p_array* header = clib.table_get_header(_table_name)
    cdef int i
    if header is NULL:
        raise ValueError("No summary for table: {!r}".format(table_name))
    return dict([_split_header_line(header.p[i])
                 for i in xrange(header.curr)])


def get_table_column_names(table_name):
    """
    Get a list of all column names in the table.

    :param str table_name: table name
    :returns: column names
    :rtype: list
    :raises ValueError: if the table name is invalid
    """
    cdef bytes _table_name = _cstr(table_name)
    cdef int index = clib.name_list_pos(_table_name, clib.table_register.names)
    if index == -1:
        raise ValueError("Invalid table: {!r}".format(table_name))
    # NOTE: we can't enforce lower-case on the column names here, since this
    # breaks their usage with get_table_column():
    return _name_list(clib.table_register.tables[index].columns)


def get_table_column_count(table_name):
    """
    Get a number of columns in the table.

    :param str table_name: table name
    :returns: number of columns
    :rtype: int
    :raises ValueError: if the table name is invalid
    """
    cdef bytes _table_name = _cstr(table_name)
    cdef int index = clib.name_list_pos(_table_name, clib.table_register.names)
    if index == -1:
        raise ValueError("Invalid table: {!r}".format(table_name))
    return clib.table_register.tables[index].columns.curr


def get_table_column(table_name, column_name):
    """
    Get data from the specified table.

    :param str table_name: table name
    :param str column_name: column name
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
    cdef bytes _tab_name = _cstr(table_name)
    cdef bytes _col_name = _cstr(column_name)
    cdef clib.column_info info
    # The following snippet imitates the table_get_column function from the
    # C API of the MAD-X library. The replacement is needed for proper
    # handling of integer columns for as long as we support versions of
    # MAD-X before r4777. When this is not the case anymore, you should be
    # able to safely revert the commit that introduced this snippet:
    cdef int tab_i = clib.name_list_pos(_tab_name, clib.table_register.names)
    if tab_i == -1:
        raise ValueError("Invalid table: {!r}".format(table_name))
    cdef clib.table* _table = clib.table_register.tables[tab_i]
    cdef int col_i = clib.name_list_pos(_col_name, _table.columns)
    if col_i == -1:
        raise ValueError("Invalid column: {!r}".format(column_name))
    info.length = _table.curr
    cdef int inform = _table.columns.inform[col_i]
    if inform == clib.PARAM_TYPE_INTEGER:
        # YES, integers are internally stored as doubles in MAD-X:
        info.data = _table.d_cols[col_i]
        info.datatype = b'i'
    elif inform == clib.PARAM_TYPE_DOUBLE:
        info.data = _table.d_cols[col_i]
        info.datatype = b'd'
    elif inform == clib.PARAM_TYPE_STRING:
        info.data = _table.s_cols[col_i]
        info.datatype = b'S'
    else:
        raise RuntimeError("Unknown column format: {!r}".format(inform))
    # Although, we are using a custom replacement for table_get_column
    # above, we still try to be fully compatible with the real function from
    # MAD-X below, so back-migration will be easier, when the time comes.
    # This is why the error and data type handling below is left untouched:
    dtype = <bytes> info.datatype
    size = <int> info.length
    addr = <Py_intptr_t> info.data
    # double:
    if dtype == b'i' or dtype == b'd':
        # YES, integers are internally stored as doubles in MAD-X:
        array_type = ctypes.c_double * size
        array_data = array_type.from_address(addr)
        return np.ctypeslib.as_array(array_data)
    # string:
    elif dtype == b'S':
        char_tmp = <char**> info.data
        return np.array([char_tmp[i] for i in xrange(info.length)])
    # invalid:
    elif dtype == b'V':
        raise ValueError("Column {!r} is not in table {!r}."
                         .format(column_name, table_name))
    # unknown:
    else:
        raise RuntimeError("Unknown datatype {!r} in column {!r}."
                           .format(_str(dtype), column_name))


def get_element(sequence_name, element_index):
    """
    Return requested element in the original sequence.

    :param str sequence_name: sequence name
    :param int element_index: element index
    :returns: the element with the specified index
    :rtype: dict
    :raises ValueError: if the sequence is invalid
    :raises IndexError: if the index is out of range
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if element_index < 0 or element_index >= seq.nodes.curr:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(element_index, seq.nodes.curr))
    return _get_node(seq.nodes.nodes[element_index], seq.ref_flag)


def get_element_index(sequence_name, element_name):
    """
    Return index of element with specified name in the original sequence.

    :param str sequence_name: sequence name
    :param str element_name: element index
    :returns: the index of the specified element
    :rtype: int
    :raises ValueError: if the sequence or element name is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef bytes _element_name = _cstr(element_name)
    cdef int index = clib.name_list_pos(_element_name, seq.nodes.list)
    if index == -1:
        raise ValueError("Element name not found: {0}".format(element_name))
    return index


def get_element_index_by_position(sequence_name, position):
    """
    Return index of element at specified position in the original sequence.

    :param str sequence_name: sequence name
    :param double position: position (S coordinate)
    :returns: the index of an element at that position, -1 if not found
    :rtype: int
    :raises ValueError: if the sequence or element name is invalid
    """
    # This is implemented here just for performance, but still uses a linear
    # algorithm. If you want more, you need to copy the elements over and use
    # a suitable ordered data type.
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef double _position = position
    cdef clib.node* elem
    cdef double at
    for i in xrange(seq.nodes.curr):
        elem = seq.nodes.nodes[i]
        at = _get_node_entry_pos(elem, seq.ref_flag)
        if _position >= at and _position <= at+elem.length:
            return i
    raise ValueError("No element found at position: {0}".format(position))


def get_element_count(sequence_name):
    """
    Return number of elements in the original sequence.

    :param str sequence_name: sequence name
    :returns: number of elements in the original sequence
    :rtype: int
    :raises ValueError: if the sequence is invalid.
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return seq.nodes.curr


def get_expanded_element(sequence_name, element_index):
    """
    Return requested element in the expanded sequence.

    :param str sequence_name: sequence name
    :param int element_index: element index
    :returns: the element with the specified index
    :rtype: dict
    :raises ValueError: if the sequence is invalid
    :raises IndexError: if the index is out of range
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if element_index < 0 or element_index >= seq.n_nodes:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(element_index, seq.n_nodes))
    return _get_node(seq.all_nodes[element_index], seq.ref_flag)


def get_expanded_element_index(sequence_name, element_name):
    """
    Return index of element with specified name in the expanded sequence.

    NOTE: this is the brute-force linear-time algorithm and therefore not
    recommended for frequent execution.

    :param str sequence_name: sequence name
    :param str element_name: element index
    :returns: the index of the specified element, -1 if not found
    :rtype: int
    :raises ValueError: if the sequence is invalid
    """
    # Unfortunately, there is no name_list for the expanded list. Therefore,
    # Therefore, we can only provide a linear-time lookup.
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef bytes _element_name = _cstr(element_name)
    for i in xrange(seq.n_nodes):
        if seq.all_nodes[i].name == _element_name:
            return i
    raise ValueError("Element name not found: {0}".format(element_name))


def get_expanded_element_index_by_position(sequence_name, position):
    """
    Return index of element at specified position in the expanded sequence.

    :param str sequence_name: sequence name
    :param double position: position (S coordinate)
    :returns: the index of an element at that position
    :rtype: int
    :raises ValueError: if the sequence or element name is invalid
    """
    # This is implemented here just for performance, but still uses a linear
    # algorithm. If you want more, you need to copy the elements over and use
    # a suitable ordered data type.
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef double _position = position
    cdef clib.node* elem
    cdef double at
    for i in xrange(seq.n_nodes):
        elem = seq.all_nodes[i]
        at = _get_node_entry_pos(elem, seq.ref_flag)
        if _position >= at and _position <= at+elem.length:
            return i
    raise ValueError("No element found at position: {0}".format(position))


def get_expanded_element_count(sequence_name):
    """
    Return number of elements in the expanded sequence.

    :param str sequence_name: sequence name
    :returns: number of elements in the expanded sequence
    :rtype: int
    :raises ValueError: if the sequence is invalid.
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return seq.n_nodes


def is_sequence_expanded(sequence_name):
    """
    Check whether a sequence has already been expanded.

    :param str sequence_name: sequence name
    :returns: expanded state of the sequence
    :rtype: bool
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    return seq.n_nodes > 0


def evaluate(expression):
    """
    Evaluates an expression and returns the result as double.

    :param str expression: symbolic expression to evaluate
    :returns: numeric value of the expression
    :rtype: float
    """
    # TODO: This function uses global variables as temporaries - which is in
    # general an *extremely* bad design choice. Even though MAD-X uses global
    # variables internally anyway, this is no excuse for cpymad to copy that
    # behaviour.
    try:
        # handle instance of type Expression:
        expression = expression.expr
    except AttributeError:
        pass
    # TODO: not sure about the flags (the magic constants 0, 2)
    cdef bytes _expr = _cstr(expression.lower())
    clib.pre_split(_expr, clib.c_dum, 0)
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
        # enforce lower-case keys:
        name = _str(cmd.par.parameters[i].name).lower()
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
    key = key.lower()
    if kind == "%le":
        return key, float(value)    # convert to number
    elif kind.endswith('s'):
        return key, value[1:-1]     # strip quotes from string
    else:
        return key, value           #


cdef _name_list(clib.name_list* names):
    """Return a python list of names for the name_list."""
    cdef int i
    return [names.names[i] for i in xrange(names.curr)]


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
    data = _get_element(node.p_elem)
    data.update({'name': _str(node.name),
                 'type': _str(node.base_name),
                 'at': _get_node_entry_pos(node, ref_flag)})
    return data


cdef double _get_node_entry_pos(clib.node* node, int ref_flag):
    """Normalize 'at' value to node entry."""
    if ref_flag == clib.REF_CENTER:
        return node.at_value - node.length / 2
    elif ref_flag == clib.REF_EXIT:
        return node.at_value - node.length
    else:
        return node.at_value


cdef _get_element(clib.element* elem):
    """Return dictionary with element attributes."""
    return _parse_command(elem.def_)
