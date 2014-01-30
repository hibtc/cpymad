#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------

# This file holds the table struct as defined in madx.h
#
# In the current state, this doesn't work very well,
# segfaults more often than not, and isn't helpful..
#

cdef extern from "madX/mad_def.h":
    enum:
        NAME_L

cdef extern from "madX/madx.h":
    struct char_p_array:
        int flag,stamp
        char** p
        char[NAME_L] name
        int  max                     # max. array size
        int  curr                   # current occupation
        int* i

    struct char_array:      # dynamic array of char
        int stamp
        int max                      # max. array size
        int curr                     # current occupation
        char* c

    struct int_array:
        char* name
        int curr
        int *i

    struct node:
        pass

    struct name_list:
        char[NAME_L]  name
        int  max                      # max. pointer array size
        int  curr                     # current occupation

cdef extern from "madX/mad_table.h":
    cdef struct table:
        char* name
        int num_cols, org_cols,dynamic,origin,curr
        char_p_array *header #,*node_nm
        int_array *col_out,*row_out
        name_list* columns    #names + types (in inform)
        char ***s_cols

cdef extern from "madX/madx.h":
    # to be able to read sequence information..
    struct sequence:
        char[NAME_L] name
        table* tw_table       #pointer to latest twiss table created

    # list of sequences..
    struct sequence_list:
        sequence_list *list       # index list of names
        sequence **sequs      # sequence pointer list
        int curr

    cdef struct column_info:
        void * data
        int length
        char datatype
        char datasize

cdef extern from "madX/mad_expr.h":
    struct expression:
        pass


cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr)
