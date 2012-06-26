from cern.pymad.domain.tfs import TfsTable,TfsSummary

from cern.libmadx.madx_structures cimport column_info
cdef extern from "madX/mad_table.h":
    column_info  table_get_column(char* table_name,char* column_name)

from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.
cdef class ArrayWrapper:

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
        #self.dtype=dtype

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


def get_dict_from_mem(table,columns,retdict):
    ret={}
    cdef column_info info
    cdef np.ndarray _tmp

    if type(columns)==str:
        columns=columns.split(',')
    
    for c in columns:
        info=table_get_column(table,c)
        dtype=<bytes>info.datatype
        if dtype=='d':
            aw=ArrayWrapper()
            aw.set_data(info.length,info.data)
            _tmp = np.array(aw, copy=False)
            # Assign our object to the 'base' of the ndarray object
            _tmp.base = <PyObject*> aw
            Py_INCREF(aw)
            ret[c.lower()]=_tmp
        elif dtype=='S':
            print "String, skipping for now..",c
        elif dtype=='V':
            print "ERROR:",c,"is not available in table",table
        else:
            print "Unknown datatype",dtype,c
    if retdict:
        return ret,{}
    return TfsTable(ret),TfsSummary({})
