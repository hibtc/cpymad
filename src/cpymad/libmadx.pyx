# cython: embedsignature=True
# cython: language_level=3
"""
Low level cython binding to MAD-X.

CAUTION: Do not import this module directly! Use :class:`Madx` instead.

- The API of this module is considered private, i.e. it may change between
  versions without further notice.
- Importing this module means loading MAD-X directly into the process space.
  This means that any crash in the (sometimes fragile) MAD-X interpreter will
  crash the importing process with it.
- All functions in this module operate on a shared global state.
- this module exposes a very C-ish API that is not convenient to work with.
"""

from os import getcwd

import ctypes
import numpy as np      # Import the Python-level symbols of numpy

# Import a large-enough integer type to hold pointer, see also:
# http://grokbase.com/t/gg/cython-users/134b21rga8/passing-callback-pointers-to-python-and-back
cdef extern from "pyport.h":
    ctypedef int Py_intptr_t

cdef extern from "string.h" nogil:
    char* strstr(char* s1, char* s2)
    char* strchr(char* s, int c)

from cpymad.types import Constraint, Parameter, AlignError, FieldError, PhaseError
from cpymad.util import name_to_internal, name_from_internal, normalize_range_name
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
    'eval',

    # Globals
    'get_var',
    'num_globals',
    'get_globals',
    'get_var_type',

    'get_options',

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
    'get_table_column_count',
    'get_table_column',
    'get_table_row',
    'get_table_row_count',
    'get_table_row_names',

    'get_table_selected_rows',
    'apply_table_selections',

    # sequence element list access
    'get_element',
    'get_element_name',
    'get_element_positions',
    'get_element_names',
    'get_element_index',
    'get_element_index_by_position',
    'get_element_count',

    # expanded sequence element access
    'get_expanded_element',
    'get_expanded_element_name',
    'get_expanded_element_positions',
    'get_expanded_element_names',
    'get_expanded_element_index',
    'get_expanded_element_index_by_position',
    'get_expanded_element_count',

    # global elements
    'get_global_element',
    'get_global_element_name',
    'get_global_element_index',
    'get_global_element_count',

    # element base types
    'get_base_type_names',

    # defined commands
    'get_defined_command',
    'get_defined_command_names',

    # imported from 'os' for convenience in madx.Madx and should not be
    # considered part of the public interface:
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
    :returns: success status, whether the command has completed without error
    :rtype: bool
    """
    cmd = cmd.rstrip().rstrip(';') + ';'
    cdef bytes _cmd = _cstr(cmd)
    cdef char* _pch = _cmd
    cdef int error = 0
    with nogil:
        _strip_comments(_pch)
        clib.supp_lt(_pch, 0)
        clib.stolower_nq(_pch)
        clib.pro_input(_pch)
        error = clib.geterrorflag()
        clib.clearerrorflag()
    return not error


def get_var(name):
    """
    Get the value of a global variable.
    """
    cdef clib.variable* var = _get_var(name)
    cdef int index = clib.name_list_pos(var.name, clib.variable_list.list)
    cdef int inform = index >= clib.start_var
    if var.type == clib.VAR_TYPE_STRING:
        return Parameter(
            name, _str(var.string), None,
            dtype=clib.PARAM_TYPE_STRING, inform=inform, var_type=var.type)
    cdef double value = clib.variable_value(var)
    cdef int typeid
    if var.val_type == clib.VAL_TYPE_INTEGER:
        typeid = clib.PARAM_TYPE_INTEGER
    else:
        typeid = clib.PARAM_TYPE_DOUBLE
    return Parameter(
        name, *_expr(var.expr, value, typeid),
        dtype=typeid, inform=inform, var_type=var.type)


def get_var_type(name):
    """
    Get the type of the variable:

        0   constant
        1   direct
        2   deferred
        3   string
    """
    return _get_var(name).type


def get_options():
    """Get the current option values."""
    return _parse_command(clib.options)


def num_globals():
    """
    Return the number of global variables.
    """
    return clib.variable_list.curr


def get_globals():
    """
    Get a list of names of all global variables.
    """
    return _name_list(clib.variable_list.list)


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
        raise RuntimeError("TWISS table invalid for {!r}".format(sequence_name))
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
        raise RuntimeError("No beam attached to {!r}".format(sequence_name))
    return _parse_command(seq.beam)


def get_sequence_length(sequence_name):
    """
    Get the length associated to the sequence.

    :param str sequence_name: sequence name
    :returns: sequence length
    :rtype: Parameter
    :raises ValueError: if the sequence name is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    length = Parameter(
            'length',
            *_expr(seq.l_expr, seq.length),
            dtype=clib.PARAM_TYPE_DOUBLE,
            inform=1)
    return length

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
            for i in range(seqs.curr)]


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
            for i in range(clib.table_register.names.curr)]


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
                 for i in range(header.curr)])


def get_table_column_names(table_name, selected=False):
    """
    Get a list of all column names in the table.

    :param str table_name: table name
    :param bool selected: consider only selected columns
    :returns: column names
    :rtype: list
    :raises ValueError: if the table name is invalid
    """
    cdef clib.table* table = _find_table(table_name)
    # NOTE: we can't enforce lower-case on the column names here, since this
    # breaks their usage with get_table_column():
    if selected:
        indices = [table.col_out.i[i] for i in range(table.col_out.curr)]
        return [_str(table.columns.names[i]) for i in indices]
    else:
        return _name_list(table.columns)


def get_table_column_count(table_name, selected=False):
    """
    Get a number of columns in the table.

    :param str table_name: table name
    :param bool selected: consider only selected columns
    :returns: number of columns
    :rtype: int
    :raises ValueError: if the table name is invalid
    """
    cdef clib.table* table = _find_table(table_name)
    if selected:
        return table.col_out.curr
    else:
        return table.columns.curr


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
    cdef clib.column_info info = clib.table_get_column(_tab_name, _col_name)
    dtype = <bytes> info.datatype
    # double:
    if dtype == b'i' or dtype == b'd':
        # YES, integers are internally stored as doubles in MAD-X:
        return np.ctypeslib.as_array(<double [:info.length]> info.data)
    # string:
    elif dtype == b'S':
        char_tmp = <char**> info.data
        return np.array([_str(char_tmp[i]) for i in range(info.length)])
    # invalid:
    elif dtype == b'V':
        raise ValueError("Column {!r} is not in table {!r}."
                         .format(column_name, table_name))
    # unknown:
    else:
        raise RuntimeError("Unknown datatype {!r} in column {!r}."
                           .format(_str(dtype), column_name))


def get_table_row(table_name, row_index, columns='all'):
    """
    Return row as tuple of values.
    """
    cdef clib.table* table = _find_table(table_name)

    if row_index < -table.curr or row_index >= table.curr:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(row_index, table.curr))
    if row_index < 0:
        row_index += table.curr

    def fetch_value(col_index):
        inform = table.columns.inform[col_index]
        if inform == clib.PARAM_TYPE_INTEGER:
            return int(table.d_cols[col_index][row_index])
        elif inform == clib.PARAM_TYPE_DOUBLE:
            return table.d_cols[col_index][row_index]
        elif inform == clib.PARAM_TYPE_STRING:
            return _str(table.s_cols[col_index][row_index])
        raise RuntimeError("Unknown datatype {!r} in column {!r}."
                           .format(inform, table.columns.names[col_index]))

    if columns == 'all':
        indices = range(table.columns.curr)
    elif columns == 'selected':
        indices = [table.col_out.i[i] for i in range(table.col_out.curr)]
        # fallback if selection missing:
        indices = indices or range(table.columns.curr)
    else:
        indices = []
        for col in columns:
            if isinstance(col, int):
                if col < 0 or col > table.columns.curr:
                    raise IndexError(
                        "Column index out of range: {0} (column count is {1})"
                        .format(col, table.columns.curr))
                indices.append(col)
            else:
                idx = clib.name_list_pos(_cstr(col), table.columns)
                if idx < 0:
                    raise ValueError("Column not found: {0!r}".format(col))
                indices.append(idx)

    return {
        _str(table.columns.names[i]): fetch_value(i)
        for i in indices
    }


def get_table_row_count(table_name):
    """
    Return total number of rows in the table.
    """
    return _find_table(table_name).curr


def get_table_row_names(table_name, indices=None):
    """
    Return row names for every index (row number) in the list.
    """
    cdef clib.table* table = _find_table(table_name)
    if indices == 'all' or indices is None:
        indices = range(table.curr)
    elif indices == 'selected':
        indices = [table.row_out.i[i] for i in range(table.curr)]
    elif isinstance(indices, int):
        return _get_table_row_name(table, indices)
    return [_get_table_row_name(table, i) for i in indices]


def get_table_selected_rows(table_name):
    """Return list of selected row indices in table (may be empty)."""
    cdef clib.table* table = _find_table(table_name)
    return [table.row_out.i[i] for i in range(table.curr)]


def apply_table_selections(table_name):
    """
    Apply the SELECT/DESELECT commands for table columns/rows.

    Needed as replacement for the missing ``out_table`` call for initializing
    ``t.row_out``, ``t.col_out`` if the twiss command was performed without a
    filename.
    """
    cdef clib.table* t = _find_table(table_name)
    version_info = tuple(map(int, get_version_number().split('.')))
    if version_info > (5,3,7):       # will crash before
        clib.out_table(_cstr(table_name), t, NULL)


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
    return _get_node(seq.nodes.nodes[element_index], seq.ref_flag, seq.n_nodes > 0, seq.line)


def get_element_positions(sequence_name):
    """
    Get list with positions of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :returns: positions of all elements in the sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef int i
    cdef clib.node** nodes = seq.nodes.nodes
    return [_get_node_entry_pos(nodes[i], seq.ref_flag, seq.n_nodes > 0)
            for i in range(seq.nodes.curr)]

def get_element_names(sequence_name):
    """
    Get list with the names of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :returns: names of all elements in the sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef int i
    return [_node_name(seq.nodes.nodes[i])
            for i in range(seq.nodes.curr)]


def get_element_name(sequence_name, element_index):
    """
    Get list with the names of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :param int element_index: element index
    :returns: the name of the element with the specified index
    :rtype: str
    :raises ValueError: if the sequence is invalid
    :raises IndexError: if the index is out of range
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if element_index < 0 or element_index >= seq.nodes.curr:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(element_index, seq.nodes.curr))
    return _node_name(seq.nodes.nodes[element_index])


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
    cdef bytes _element_name = _cstr(name_to_internal(element_name))
    cdef int index = clib.name_list_pos(_element_name, seq.nodes.list)
    if index == -1:
        raise ValueError("Element name not found: {0!r}".format(element_name))
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
    for i in range(seq.nodes.curr):
        elem = seq.nodes.nodes[i]
        at = _get_node_entry_pos(elem, seq.ref_flag, seq.n_nodes > 0)
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

    NOTE: this function may currently return elements beyond the end of the
    expanded sequence if requested to do so.
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if element_index < 0 or element_index > seq.n_nodes:
        raise IndexError(
            "Index out of range: {0} (element count is {1})"
            .format(element_index, seq.n_nodes))
    return _get_node(seq.all_nodes[element_index], seq.ref_flag, seq.n_nodes > 0, seq.line)


def get_expanded_element_positions(sequence_name):
    """
    Get list with positions of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :returns: positions of all elements in the sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef int i
    cdef clib.node** nodes = seq.all_nodes
    return [_get_node_entry_pos(nodes[i], seq.ref_flag, seq.n_nodes > 0)
            for i in range(seq.n_nodes)]


def get_expanded_element_names(sequence_name):
    """
    Get list with the names of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :returns: names of all elements in the sequence
    :rtype: list
    :raises ValueError: if the sequence is invalid
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    cdef int i
    return [_node_name(seq.all_nodes[i])
            for i in range(seq.n_nodes)]


def get_expanded_element_name(sequence_name, element_index):
    """
    Get list with the names of all elements of a specific sequence.

    :param str sequence_name: sequence name
    :param int element_index: element index
    :returns: the name of the element with the specified index
    :rtype: str
    :raises ValueError: if the sequence is invalid
    :raises IndexError: if the index is out of range
    """
    cdef clib.sequence* seq = _find_sequence(sequence_name)
    if element_index < 0 or element_index > seq.n_nodes:
        raise IndexError(
            "Index out of range: {0} (element count is {1})"
            .format(element_index, seq.n_nodes))
    return _node_name(seq.all_nodes[element_index])


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
    cdef bytes _element_name = _cstr(name_to_internal(element_name))
    for i in range(seq.n_nodes):
        if seq.all_nodes[i].name == _element_name:
            return i
    raise ValueError("Element name not found: {0!r}".format(element_name))


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
    for i in range(seq.n_nodes):
        elem = seq.all_nodes[i]
        at = _get_node_entry_pos(elem, seq.ref_flag, seq.n_nodes > 0)
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


def get_global_element(element_index):
    """
    Return requested element in the expanded sequence.

    :param int element_index: element index
    :returns: the element with the specified index
    :rtype: dict
    :raises IndexError: if the index is out of range
    """
    cdef clib.el_list* elems = clib.element_list
    if element_index < 0 or element_index >= elems.curr:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(element_index, elems.curr))
    return _get_element(elems.elem[element_index])


def get_global_element_name(element_index):
    """
    Return name of element.

    :param int element_index: element index
    :returns: element name
    :rtype: str
    :raises IndexError: if the index is out of range
    """
    cdef clib.el_list* elems = clib.element_list
    if element_index < 0 or element_index >= elems.curr:
        raise IndexError("Index out of range: {0} (element count is {1})"
                         .format(element_index, elems.curr))
    return _str(elems.list.names[element_index])


def get_global_element_index(element_name):
    """
    Return index of element with specified name in the global element list.

    :param str element_name: element index
    :returns: the index of the specified element, -1 if not found
    :rtype: int
    """
    cdef bytes _element_name = _cstr(element_name)
    return clib.name_list_pos(_element_name, clib.element_list.list)


def get_global_element_count():
    """
    Return number of globally visible elements.
    """
    return clib.element_list.curr


def get_base_type_names():
    """
    Return list of element names for base types.
    """
    return _name_list(clib.base_type_list.list)


def get_defined_command(command_name):
    """Return MAD-X command as dict of values."""
    cdef bytes _command_name = _cstr(command_name)
    cdef int index = clib.name_list_pos(_command_name, clib.defined_commands.list)
    if index == -1:
        raise ValueError("Invalid command: {!r}".format(command_name))
    return _parse_command(clib.defined_commands.commands[index])


def get_defined_command_names():
    """Return list of MAD-X command names."""
    return _name_list(clib.defined_commands.list)


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


def eval(expression):
    """
    Evaluates an expression and returns the result as double.

    :param str expression: symbolic expression to evaluate
    :returns: numeric value of the expression
    :rtype: float

    NOTE: This function does not perform rigorous input validation! It uses
    nothing but the MAD-X builtin rather incompetent error checks. This means
    invalid input such as '+' can lead to program crashes! If you're looking
    for more secure validation, see :func:`cpymad.util.check_expression`.
    """
    cdef clib.expression* expr = _make_expr(expression)
    if expr == NULL:
        raise ValueError("Invalid expression: {!r}".format(expression))
    value = clib.expression_value(expr, 2)
    clib.delete_expression(expr)
    return value


def expression_vars(expression):
    """
    Returns all the variables in an expression.

    :param str expression: symbolic expression
    :returns: set of variables
    :rtype: set

    NOTE: This function does not perform rigorous input validation! It uses
    nothing but the MAD-X builtin rather incompetent error checks. This means
    invalid input such as '+' can lead to program crashes! If you're looking
    for more secure validation, see :func:`cpymad.util.check_expression`.
    """
    cdef clib.expression* expr = _make_expr(expression)
    if expr == NULL:
        raise ValueError("Invalid expression: {!r}".format(expression))
    cdef int i
    vars = {_str(clib.expr_chunks.names[k % 100000000])
            for i in range(expr.polish.curr)
            for k in [expr.polish.i[i]]
            if k // 100000000 == 1}
    clib.delete_expression(expr)
    return vars


# Helper functions:

# The following functions are `cdef functions`, i.e. they can only be
# called from Cython code. It is necessary to use `cdef functions` whenever
# we want to pass parameters or return values with a pure C type.


cdef clib.expression* _make_expr(expression):
    # TODO: This function uses global variables as temporaries - which is in
    # general an *extremely* bad design choice. Even though MAD-X uses global
    # variables internally anyway, this is no excuse for cpymad to copy that
    # behaviour.
    # TODO: not sure about the flags (the magic constants 0, 2)
    cdef bytes _expr = _cstr(expression.lower())
    clib.pre_split(_expr, clib.c_dum, 0)
    clib.mysplit(clib.c_dum.c, clib.tmp_p_array)
    # NOTE: `loc_expr` is mostly useless for input validation. It even accepts
    # stuff such as '+' that will lead to program crashes on evaluation. We
    # use it nevertheless to get a minimal amount of error-checking. Not using
    # cpymad.util.check_expression for simplicity and performance.
    cdef int start=0, stop
    etype = clib.loc_expr(clib.tmp_p_array.p,
                          clib.tmp_p_array.curr,
                          start, &stop)
    if etype == 0 or stop+1 < clib.tmp_p_array.curr:
        return NULL
    return clib.make_expression(clib.tmp_p_array.curr, clib.tmp_p_array.p)


_expr_types = [bool, int, float]

cdef _expr(clib.expression* expr,
           double value,
           int typeid=clib.PARAM_TYPE_DOUBLE):
    """Return a parameter value with an appropriate type."""
    _type = _expr_types[typeid]
    _value = _type(value)
    if expr is NULL or expr.string is NULL:
        return _value, None
    _expr = _str(expr.string).strip()
    if not _expr:
        return _value, None
    try:
        return _type(float(_expr)), None
    except ValueError:
        pass
    value = clib.expression_value(expr, 2)
    return _type(value), _expr


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
        return _str(par.string), None

    if par.type == clib.PARAM_TYPE_CONSTRAINT:
        val = min = max = (None, None)
        if par.c_type == clib.CONSTR_TYPE_NONE: # occurs in defined_commands
            pass
        if par.c_type == clib.CONSTR_TYPE_MIN:
            min = _expr(par.min_expr, par.c_min)
        if par.c_type == clib.CONSTR_TYPE_MAX:
            max = _expr(par.max_expr, par.c_max)
        if par.c_type == clib.CONSTR_TYPE_BOTH:
            min = _expr(par.min_expr, par.c_min)
            max = _expr(par.max_expr, par.c_max)
        if par.c_type == clib.CONSTR_TYPE_VALUE:
            val = _expr(par.expr, par.double_value)
        return (Constraint(val=val[0], min=min[0], max=max[0]),
                Constraint(val=val[1], min=min[1], max=max[1]))

    if par.type in (clib.PARAM_TYPE_INTEGER_ARRAY, clib.PARAM_TYPE_DOUBLE_ARRAY):
        fields = [
            _expr(NULL if par.expr_list is NULL else par.expr_list.list[i],
                  par.double_array.a[i],
                  par.type - clib.PARAM_TYPE_LOGICAL_ARRAY)
            for i in range(par.double_array.curr)
        ]
        return ([f[0] for f in fields],
                [f[1] for f in fields])

    if par.type == clib.PARAM_TYPE_STRING_ARRAY:
        return [_str(par.m_string.p[i])
                for i in range(par.m_string.curr)], None

    raise ValueError("Unknown parameter type: {}".format(par.type))


cdef _parse_command(clib.command* cmd):
    """
    Get the values of all parameters of a command.

    :returns: the command parameters
    :rtype: dict
    """
    cdef int i
    return {'name': _str(cmd.name), 'data': {
        # enforce lower-case keys:
        name: Parameter(
            name,
            *_get_param_value(cmd.par.parameters[i]),
            dtype=cmd.par.parameters[i].type,
            inform=cmd.par_names.inform[i])
        for i in range(cmd.par.curr)
        for name in [_str(cmd.par.parameters[i].name).lower()]
    }}


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
        raise ValueError("Invalid sequence: {!r}".format(sequence_name))
    return seqs.sequs[index]


cdef clib.table* _find_table(table_name) except NULL:
    cdef bytes _table_name = _cstr(table_name)
    cdef int index = clib.name_list_pos(_table_name, clib.table_register.names)
    if index == -1:
        raise ValueError("Invalid table: {!r}".format(table_name))
    return clib.table_register.tables[index]


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
    return [_str(names.names[i]) for i in range(names.curr)]


# Leave return type unspecified to avoid cython error with unicode.split(None):
# ("TypeError: coercing to Unicode: need string or buffer, NoneType found")
cdef _str(const char* s):
    """Decode C string to python string."""
    if s is NULL:
        return ""
    return s.decode('utf-8')


cdef bytes _cstr(s):
    """Encode python string to C string."""
    if s is None:
        return b""
    if isinstance(s, bytes):
        return <bytes> s
    return <bytes> s.encode('utf-8')


cdef _node_name(clib.node* node):
    return name_from_internal(_str(node.name))


cdef _get_node(clib.node* node, int ref_flag, int is_expanded, int line):
    """Return dictionary with node + element attributes."""
    if node.p_elem is NULL:
        # Maybe this is a valid case, but better detect it with boom!
        raise RuntimeError("Empty node or subsequence! Please report this incident!")
    data = _get_element(node.p_elem)
    if not line:
        # Update `at` command parameters in order to avoid surprises when
        # specifying `at` value for elements defined outside the sequence:
        data['data']['at'] = Parameter(
            'at', *_expr(node.at_expr, node.at_value),
            dtype=clib.PARAM_TYPE_DOUBLE, inform=1)
        data['data']['from'] = Parameter(
            'from', _str(node.from_name), None,
            dtype=clib.PARAM_TYPE_STRING, inform=node.from_name != NULL)
    data['node_name'] = _node_name(node)
    data['occ_cnt'] = node.occ_cnt
    data['enable'] = node.enable
    data['base_name'] = _str(node.base_name)
    data['position'] = _get_node_entry_pos(node, ref_flag, is_expanded)
    data['length'] = node.length
    data['align_errors'] = None if node.p_al_err is NULL else AlignError(
        *_memview(node.p_al_err))
    data['field_errors'] = None if node.p_fd_err is NULL else FieldError(
        dkn=list(_memview(node.p_fd_err)[0::2]),
        dks=list(_memview(node.p_fd_err)[1::2]))
    data['phase_errors'] = None if node.p_ph_err is NULL else PhaseError(
        dpn=list(_memview(node.p_ph_err)[0::2]),
        dps=list(_memview(node.p_ph_err)[1::2]))
    return data


cdef double [:] _memview(clib.double_array* array):
    return <double [:array.curr]> array.a


cdef double _get_node_entry_pos(clib.node* node, int ref_flag, int is_expanded):
    """Normalize 'at' value to node entry."""
    if is_expanded:
        return node.position - node.length / 2
    cdef double position = node.at_value
    if ref_flag == clib.REF_CENTER:
        return position - node.length / 2
    elif ref_flag == clib.REF_EXIT:
        return position - node.length
    else:
        return position


cdef _get_element(clib.element* elem):
    """Return dictionary with element attributes."""
    data = _parse_command(elem.def_)
    data['name'] = _str(elem.name)
    data['length'] = elem.length
    data['parent'] = _str(elem.parent.name)
    data['base_type'] = _str(elem.base_type.name)
    return data


cdef _get_table_row_name(clib.table* table, int index):
    if index < 0 or index >= table.curr:
        raise ValueError("Invalid row index: {}".format(index))
    return normalize_range_name(name_from_internal(_str(
        table.node_nm.p[index]
    )))


cdef clib.variable* _get_var(name) except NULL:
    cdef bytes _name = _cstr(name.lower())
    cdef clib.variable* var = clib.find_variable(_name, clib.variable_list)
    if var is NULL:
        raise KeyError("Variable not defined: {!r}".format(name))
    return var


cdef void _strip_comments(char* text) nogil:
    cdef char* dest = text
    cdef char c, d
    while True:
        c = text[0]
        if c == 0:
            break
        d = text[1]
        if c == b'!' or (c == b'/' and d == b'/'):
            text = strchr(text+1, b'\n')
            if text == NULL:
                break
            continue
        if c == b'/' and d == b'*':
            text = strstr(text+2, b"*/")
            if text == NULL:
                break
            text += 2
            continue
        dest[0] = text[0]
        text += 1
        dest += 1
    dest[0] = 0
