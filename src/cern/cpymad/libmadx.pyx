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

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

import numpy as np      # Import the Python-level symbols of numpy
cimport numpy as np     # Import the C-level symbols of numpy


# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef _split_header_line(header_line):
    header_line = header_line.decode('utf-8')
    hsplit=header_line.split()
    if len(hsplit)!=4 or hsplit[0]!='@':
        raise ValueError("Could not read header line: %s" % header_line)
    key=hsplit[1]
    value=hsplit[3]
    if hsplit[2]=="%le":
        value=float(value)
    return key,value

def get_table(table, columns):
    """
    Get data from the specified tables.

    :param str table: table name
    :param list columns: column names
    :returns: the data in the requested columns
    :rtype: dict

    CAUTION: Numeric data is wrapped in numpy arrays but not copied. Make
    sure to copy all data before invoking any further MAD-X commands!

    """
    ret={}
    cdef column_info info
    cdef char_p_array *header
    cdef char** char_tmp
    cdef np.npy_intp shape[1]

    # reading the header information..
    table = table.encode('utf-8')
    header = <char_p_array*>table_get_header(table)
    ret_header={}
    for i in xrange(header.curr):
        key,value=_split_header_line(header.p[i])
        ret_header[key]=value

    # reading the columns that were requested..
    for c in columns:
        col_bytes = c.encode('utf-8')
        info=table_get_column(table,col_bytes)
        dtype = <bytes>info.datatype
        if dtype==b'd':
            shape[0] = <np.npy_intp> info.length
            ret[c.lower()] = np.PyArray_SimpleNewFromData(
                1, shape, np.NPY_DOUBLE, info.data)
        elif dtype==b'S':
            char_tmp=<char**>info.data
            ret[c.lower()]=np.zeros(info.length,'S%d'%info.datasize)
            for i in xrange(info.length):
                ret[c.lower()][i]=char_tmp[i]
        elif dtype==b'V':
            print("ERROR:",c,"is not available in table",table)
        else:
            print("Unknown datatype",dtype,c)

    return ret,ret_header


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

