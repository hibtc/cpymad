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
from __future__ import absolute_import
from __future__ import print_function

from cern.pymad.domain.tfs import TfsTable,TfsSummary

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

import numpy as np      # Import the Python-level symbols of numpy
cimport numpy as np     # Import the C-level symbols of numpy

cimport cern.cpymad.libmadx as libmadx



# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.
cdef class ArrayWrapper:
    dtype=np.NPY_DOUBLE

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Parameters:
        -----------
        size: int
        Length of the array.
        data_ptr: void*
        Pointer to the data

        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
        tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_DOUBLE, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """
        Frees the array. This is called by Python when all the
        references to the object are gone.

        Since we are using the memory which Mad-X might need
        later on, let's not.
        """
        pass

cdef class ArrayWrapperInt(ArrayWrapper):
    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
        tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_INT, self.data_ptr)
        return ndarray


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

def get_dict_from_mem(table,columns,retdict):
    ret={}
    cdef libmadx.column_info info
    cdef libmadx.char_p_array *header
    cdef np.ndarray _tmp
    cdef char** char_tmp
    if type(columns)==str:
        columns=columns.split(',')


    # reading the header information..
    table = table.encode('utf-8')
    header = <libmadx.char_p_array*>table_get_header(table)
    ret_header={}
    for i in xrange(header.curr):
        key,value=_split_header_line(header.p[i])
        ret_header[key]=value



    # reading the columns that were requested..
    for c in columns:
        col_bytes = c.encode('utf-8')
        info=table_get_column(table,col_bytes)
        dtype=<bytes>info.datatype
        if dtype==b'd':
            aw=ArrayWrapper()
            aw.set_data(info.length,info.data)
            _tmp = np.array(aw, copy=False)
            # Assign our object to the 'base' of the ndarray object
            _tmp.base = <PyObject*> aw
            Py_INCREF(aw)
            ret[c.lower()]=_tmp
        elif dtype==b'S':
            char_tmp=<char**>info.data
            ret[c.lower()]=np.zeros(info.length,'S%d'%info.datasize)
            for i in xrange(info.length):
                ret[c.lower()][i]=char_tmp[i]
        elif dtype==b'V':
            print("ERROR:",c,"is not available in table",table)
        else:
            print("Unknown datatype",dtype,c)

    if retdict:
        return ret,ret_header
    return TfsTable(ret),TfsSummary(ret_header)


# Python-level binding to libmadx:

def start():
    """
    Initialize MAD-X.
    """
    libmadx.madx_start()

def finish():
    """
    Cleanup MAD-X.
    """
    libmadx.madx_finish()

def input(cmd):
    """
    Pass one input command to MAD-X.

    :param str cmd: command to be executed by the MAD-X interpretor

    """
    cmd = cmd.encode('utf-8')
    cdef char* _cmd = cmd
    libmadx.stolower_nq(_cmd)
    libmadx.pro_input(_cmd)

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
    cdef libmadx.sequence_list *seqs
    seqs= libmadx.madextern_get_sequence_list()
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
    libmadx.pre_split(cmd, libmadx.c_dum, 0)
    libmadx.mysplit(libmadx.c_dum.c, libmadx.tmp_p_array)
    expr = libmadx.make_expression(libmadx.tmp_p_array.curr, libmadx.tmp_p_array.p)
    value = libmadx.expression_value(expr, 2)
    libmadx.delete_expression(expr)
    return value

