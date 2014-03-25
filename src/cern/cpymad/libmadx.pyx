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
import numpy as np      # Import the Python-level symbols of numpy
cimport numpy as np     # Import the C-level symbols of numpy

from cern.cpymad.types import Constraint, Expression


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


# Python-level binding to libmadx:

def start():
    """
    Initialize MAD-X.
    """
    madx_start()


def finish():
    """
    Cleanup MAD-X.
    """
    madx_finish()


def input(cmd):
    """
    Pass one input command to MAD-X.

    :param str cmd: command to be executed by the MAD-X interpretor
    """
    cmd = cmd.encode('utf-8')
    cdef char* _cmd = cmd
    stolower_nq(_cmd)
    pro_input(_cmd)


def get_twiss(sequence_name):
    """
    Get the last calculated twiss table for the given sequence.

    :param str sequence_name: sequence name
    :returns: twiss table name
    :rtype: str
    :raises ValueError: if the sequence name is invalid
    :raises RuntimeError: if the twiss table is invalid
    """
    cdef sequence* seq
    seq = _find_sequence(sequence_name)
    if not seq.tw_valid:
        raise RuntimeError("TWISS table invalid.")
    return seq.tw_table.name.decode('utf-8')


def get_beam(sequence_name):
    """
    Get the beam associated to the sequence.

    :param str sequence_name: sequence name
    :returns: beam properties as set with the BEAM command
    :rtype: dict
    :raises ValueError: if the sequence name is invalid
    :raises RuntimeError: if the sequence has no associated beam
    """
    cdef sequence* seq
    seq = _find_sequence(sequence_name)
    if seq.beam is NULL or not seq.beam.beam_def:
        raise RuntimeError("No beam attached to {}".format(sequence_name))
    return _parse_command(seq.beam)


def get_current_sequence():
    """
    Get the name of the active sequence.

    :returns: name of current sequence
    :rtype: str
    :raises RuntimeError: if no sequence is activated
    """
    if current_sequ is NULL:
        raise RuntimeError("No active sequence!")
    return current_sequ.name.decode('utf-8')


def get_sequences():
    """
    Get a list of all sequences currently in memory.

    :returns: sequence names
    :rtype: list
    """
    cdef sequence_list *seqs
    seqs = madextern_get_sequence_list()
    return [seqs.sequs[i].name.decode('utf-8')
            for i in xrange(seqs.curr)]


def get_table_summary(table):
    """
    Get table summary.

    :param str table: table name
    :returns: mapping of {column: value}
    :rtype: dict
    """
    cdef char_p_array *header
    ctable = table.encode('utf-8')
    header = <char_p_array*> table_get_header(ctable)
    return dict(_split_header_line(header.p[i])
                for i in xrange(header.curr))


def get_table_columns(table):
    """
    Get a list of all columns in the table.

    :param str table: table name
    :returns: column names
    :rtype: list
    :raises ValueError: if the table name is invalid
    """
    ctab = table.encode('utf-8')
    index = name_list_pos(ctab, table_register.names)
    if index == -1:
        raise ValueError("Invalid table: {!r}".format(table))
    return _name_list(table_register.tables[index].columns)


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
    cdef column_info info
    cdef char** char_tmp
    cdef np.npy_intp shape[1]
    ctab = table.encode('utf-8')
    ccol = column.encode('utf-8')
    info = table_get_column(ctab, ccol)
    dtype = <bytes> info.datatype
    # double:
    if dtype == b'd':
        shape[0] = <np.npy_intp> info.length
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, info.data)
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
                           .format(dtype.decode('utf-8'), column))


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
    cmd = cmd.lower().encode("utf-8")
    pre_split(cmd, c_dum, 0)
    mysplit(c_dum.c, tmp_p_array)
    expr = make_expression(tmp_p_array.curr, tmp_p_array.p)
    value = expression_value(expr, 2)
    delete_expression(expr)
    return value


# Helper functions:

# The following functions are `cdef functions`, i.e. they can only be
# called from Cython code. It is necessary to use `cdef functions` whenever
# we want to pass parameters or return values with a pure C type.

_expr_types = [bool, int, float]

cdef _expr(expression* expr, value, typeid=PARAM_TYPE_DOUBLE):
    """Return a parameter value with an appropriate type."""
    type = _expr_types[typeid]
    if expr is NULL:
        return type(value)
    else:
        return Expression(expr.string.decode('utf-8'), value, type)


cdef _get_param_value(command_parameter* par):

    """
    Get the value of a command parameter.

    :returns: value of the parameter
    :rtype: depends on the parameter type
    :raises ValueError: if the parameter type is invalid
    """

    if par.type in (PARAM_TYPE_LOGICAL,
                    PARAM_TYPE_INTEGER,
                    PARAM_TYPE_DOUBLE):
        return _expr(par.expr, par.double_value, par.type)

    if par.type == PARAM_TYPE_STRING:
        return par.string.decode('utf-8')

    if par.type == PARAM_TYPE_CONSTRAINT:
        if par.c_type == CONSTR_TYPE_MIN:
            return Constraint(min=_expr(par.min_expr, par.c_min))
        if par.c_type == CONSTR_TYPE_MAX:
            return Constraint(max=_expr(par.max_expr, par.c_max))
        if par.c_type == CONSTR_TYPE_BOTH:
            return Constraint(min=_expr(par.min_expr, par.c_min),
                              max=_expr(par.max_expr, par.c_max))
        if par.c_type == CONSTR_TYPE_VALUE:
            return Constraint(val=_expr(par.expr, par.double_value))

    if par.type in (PARAM_TYPE_INTEGER_ARRAY, PARAM_TYPE_DOUBLE_ARRAY):
        return [
            _expr(NULL if par.expr_list is NULL else par.expr_list.list[i],
                  par.double_array.a[i],
                  par.type - PARAM_TYPE_LOGICAL_ARRAY)
            for i in xrange(par.double_array.curr)
        ]

    if par.type == PARAM_TYPE_STRING_ARRAY:
        return [par.m_string.p[i].decode('utf-8')
                for i in xrange(par.m_string.curr)]

    raise ValueError("Unknown parameter type: {}".format(par.type))


cdef _parse_command(command* cmd):
    """
    Get the values of all parameters of a command.

    :returns: the command parameters
    :rtype: dict
    """
    # generator expressions are not yet supported in cdef functions, so
    # let's do it the hard way:
    res = {}
    for i in xrange(cmd.par.curr):
        name = cmd.par.parameters[i].name.decode('utf-8')
        res[name] = _get_param_value(cmd.par.parameters[i])
    return res


cdef sequence* _find_sequence(sequence_name):
    """
    Get pointer to the C sequence struct of the specified sequence or NULL.

    :param str sequence_name: sequence name
    :raises ValueError: if the sequence can not be found
    """
    cdef sequence_list *seqs
    name = sequence_name.encode('utf-8')
    seqs = madextern_get_sequence_list()
    index = name_list_pos(name, seqs.list)
    if index == -1:
        raise ValueError("Invalid sequence: {}".format(sequence_name))
    return seqs.sequs[index]


cdef _split_header_line(header_line):
    """Parse a table header value."""
    _, key, kind, value = header_line.decode('utf-8').split(None, 3)
    if kind == "%le":
        return key, float(value)    # convert to number
    elif kind.endswith('s'):
        return key, value[1:-1]     # strip quotes from string
    else:
        return key, value           # 


cdef _name_list(name_list* names):
    """Return a python list of names for the name_list."""
    return [names.names[i] for i in xrange(names.curr)]
