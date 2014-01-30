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
"""
cern.cpymad.libmadx is a Cython binding for the MAD-X library.

This file contains declarations of data structures and exported functions
of the C-API of MAD-X.

"""

# Data structures:
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

# Function declarations:
cdef extern from "madX/mad_api.h":
    sequence_list *madextern_get_sequence_list()
cdef extern from "madX/mad_core.h":
    void madx_start()
    void madx_finish()

cdef extern from "madX/mad_str.h":
    void stolower_nq(char*)
    int mysplit(char*, char_p_array*)
cdef extern from "madX/mad_eval.h":
    void pro_input(char*)

cdef extern from "madX/mad_expr.h":
    expression* make_expression(int, char**)
    double expression_value(expression*, int)
    expression* delete_expression(expression*)
cdef extern from "madX/madx.h":
    char_p_array* tmp_p_array    # temporary buffer for splits
    char_array* c_dum

cdef extern from "madX/mad_parse.h":
    void pre_split(char*, char_array*, int)

cdef extern from "madX/mad_table.h":
    column_info  table_get_column(char* table_name,char* column_name)
    char_p_array table_get_header(char* table_name)

# Utility class:
cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr)

