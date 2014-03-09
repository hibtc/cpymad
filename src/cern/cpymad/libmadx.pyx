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
from __future__ import print_function

import numpy as np      # Import the Python-level symbols of numpy
cimport numpy as np     # Import the C-level symbols of numpy

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

def get_sequences():
    '''
    Returns the sequences currently in memory

    :returns: mapping of sequence names and their twiss table names
    :rtype: dict


    This is how the return looks like in python pseudo-code:

    ..

        {s.name: {'name': s.name,
                  'twissname': s.twisstable}
         for s in sequence}

    The format of the returned data should probably be changed.

    '''
    cdef sequence_list *seqs
    seqs= madextern_get_sequence_list()
    ret={}
    for i in xrange(seqs.curr):
        name = seqs.sequs[i].name.decode('utf-8')
        ret[name]={'name':name}
        if seqs.sequs[i].tw_table.name is not NULL:
            tabname = seqs.sequs[i].tw_table.name.decode('utf-8')
            ret[name]['twissname'] = tabname
            print("Table name:", tabname)
            print("Number of columns:",seqs.sequs[i].tw_table.num_cols)
            print("Number of columns (orig):",seqs.sequs[i].tw_table.org_cols)
            print("Number of rows:",seqs.sequs[i].tw_table.curr)
    return ret

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
                for i in range(header.curr))

def get_table_column(table, column):
    """
    Get data from the specified table.

    :param str table: table name
    :param str columns: column name
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
    ccol = column.encode('utf-8')
    info = table_get_column(table, ccol)
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
        raise ValueError("Column '' is not in table ''."
                         % (column, table))
    # unknown:
    else:
        raise RuntimeError("Unknown datatype '%s' in column ''."
                           % (dtype.decode(), column))

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

# internal functions
cdef _split_header_line(header_line):
    _, key, kind, value = header_line.decode('utf-8').split(3)
    if kind == "%le":
        return key, float(value)    # convert to number
    elif kind.endswith('s'):
        return key, value[1:-1]     # strip quotes from string
    else:
        return key, value           # 

