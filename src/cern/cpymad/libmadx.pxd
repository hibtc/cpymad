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

# The following declarations are only used by the Cython compiler (so it
# knows what fields exist in the data types). The C compiler will use the
# real definitions in the header files.
# For greater clarity, we should therefore list only those fields, that are
# actually used in the Cython code and add further fields when needed
# later:

cdef extern from "madX/mad_def.h":
    enum:
        NAME_L

cdef extern from "madX/mad_array.h":
    struct char_array:
        int curr
        char* c

    struct char_p_array:
        int  curr
        char** p

    struct int_array:
        int curr
        int* i

cdef extern from "madX/mad_name.h":
    struct name_list:
        pass

cdef extern from "madX/mad_node.h":
    struct node:
        pass

cdef extern from "madX/mad_table.h":
    struct table:
        char[NAME_L] name
        int curr
        char_p_array* header
        name_list* columns

    struct column_info:
        void * data
        int length
        char datatype
        char datasize

cdef extern from "madX/mad_expr.h":
    struct expression:
        pass

cdef extern from "madX/mad_seq.h":
    struct sequence:
        char[NAME_L] name
        table* tw_table

    struct sequence_list:
        int curr
        sequence** sequs


# Global variables:
cdef extern from "madX/mad_gvar.h":
    char_p_array* tmp_p_array   # temporary buffer for splits
    char_array* c_dum           # another temporary buffer


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

cdef extern from "madX/mad_parse.h":
    void pre_split(char*, char_array*, int)

cdef extern from "madX/mad_table.h":
    column_info  table_get_column(char* table_name, char* column_name)
    char_p_array table_get_header(char* table_name)
