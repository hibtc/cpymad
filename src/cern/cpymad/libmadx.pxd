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

    struct double_array:
        int curr
        double* a

cdef extern from "madX/mad_name.h":
    struct name_list:
        int curr
        char** names

cdef extern from "madX/mad_elem.h":
    cdef struct element:
        char[NAME_L] name
        double length

cdef extern from "madX/mad_node.h":
    struct node:
        char[NAME_L] name
        node* previous # previous node
        node* next     # next node
        char* base_name
        element* p_elem # pointer to element..

cdef extern from "madX/mad_table.h":
    struct table:
        char[NAME_L] name
        int curr
        char_p_array* header
        name_list* columns

    struct table_list:
        int curr
        name_list* names
        table** tables

    struct column_info:
        void * data
        int length
        char datatype
        char datasize

cdef extern from "madX/mad_expr.h":
    struct expression:
        char* string
        double value

    struct expr_list:
        int curr
        expression** list

cdef extern from "madX/mad_cmdpar.h":
    struct command_parameter:
        char[NAME_L] name
        int type
        int c_type
        double double_value
        double c_min
        double c_max
        expression* expr
        expression* min_expr
        expression* max_expr
        char* string
        int stamp
        double_array* double_array
        expr_list* expr_list
        char_p_array* m_string

    struct command_parameter_list:
        int curr
        command_parameter** parameters

cdef enum:
    PARAM_TYPE_LOGICAL = 0
    PARAM_TYPE_INTEGER = 1
    PARAM_TYPE_DOUBLE = 2
    PARAM_TYPE_STRING = 3
    PARAM_TYPE_CONSTRAINT = 4
    PARAM_TYPE_LOGICAL_ARRAY = 10   # I invented this one for symmetry.
    PARAM_TYPE_INTEGER_ARRAY = 11
    PARAM_TYPE_DOUBLE_ARRAY = 12
    PARAM_TYPE_STRING_ARRAY = 13

cdef enum:
    CONSTR_TYPE_MIN = 1
    CONSTR_TYPE_MAX = 2
    CONSTR_TYPE_BOTH = 3
    CONSTR_TYPE_VALUE = 4

cdef extern from "madX/mad_cmd.h":
    struct command:
        int beam_def
        command_parameter_list* par

cdef extern from "madX/mad_seq.h":
    struct sequence:
        char[NAME_L] name
        command* beam
        table* tw_table
        int tw_valid
        int n_nodes
        node* start # first node..
        node* end   # last node..
        node** all_nodes

    struct sequence_list:
        int curr
        name_list* list
        sequence** sequs


# Global variables:
cdef extern from "madX/mad_gvar.h":
    sequence* current_sequ      # active sequence
    table_list* table_register  # list of all tables
    char_p_array* tmp_p_array   # temporary buffer for splits
    char_array* c_dum           # another temporary buffer


# Function declarations:
cdef extern from "madX/mad_api.h":
    sequence_list *madextern_get_sequence_list()

cdef extern from "madX/mad_core.h":
    void madx_start()
    void madx_finish()

cdef extern from "madX/mad_name.h":
    int name_list_pos(const char*, name_list*)

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
    int _table_exists "table_exists" (char* table_name)


# I have no clue why, but for some reason, it is necessary to include
# 'madx.h' (or one of the file it includes). Otherwise, importing the Cython
# module will result in an ImportError: undefined symbol: pro_input
cdef extern from "madX/madx.h":
    pass
