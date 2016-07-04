"""
cpymad.libmadx is a Cython binding for the MAD-X library.

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

cdef extern from "madX/mad_gcst.h":
    # NOTE: C API uses "const char*"
    char* version_name
    char* version_date

cdef extern from "madX/mad_array.h":
    struct char_array:
        int curr
        char* c

    struct char_p_array:
        int curr
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
        int* inform
        char** names

cdef extern from "madX/mad_table.h":
    struct table:
        char[NAME_L] name
        int curr
        char_p_array* header
        int_array* col_out      # column no.s to be written (in this order)
        char_p_array* node_nm   # names of nodes at each row
        name_list* columns

    struct table_list:
        int curr
        name_list* names
        table** tables

    struct column_info:
        void* data
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

cdef extern from "madX/mad_elem.h":
    struct element:
        char[NAME_L] name
        command* def_ "def"

    struct el_list:
        int curr                # current occupation
        name_list* list         # index list of names
        element** elem          # element pointer list

cdef extern from "madX/mad_node.h":
    struct node:
        char[NAME_L] name
        char* base_name
        double position
        double at_value
        double length
        element* p_elem

    struct node_list:
        int curr
        node** nodes
        name_list* list

cdef extern from "madX/mad_seq.h":
    struct sequence:
        # original sequence
        char[NAME_L] name
        int ref_flag
        node_list* nodes
        command* beam
        # expanded sequence
        int n_nodes
        node* ex_start          # first node in expanded sequence
        node* ex_end            # last node in expanded sequence
        node** all_nodes
        table* tw_table
        int tw_valid

    struct sequence_list:
        int curr
        name_list* list
        sequence** sequs

cdef extern from "madX/mad_var.h":
    struct variable:
        char[NAME_L] name
        int type                    # 0 constant, 1 direct, 2 deferred, 3 string
        int val_type                # 0 int 1 double (0..2)
        char* string                # pointer to string if 3
        expression* expr            # pointer to defining expression (0..2)

    struct var_list:
        int curr                    # current occupation
        name_list* list             # index list of names
        variable** vars             # variable pointer list

cdef enum:
    REF_EXIT = -1
    REF_CENTER = 0
    REF_ENTRY = 1


# Global variables:
cdef extern from "madX/mad_gvar.h":
    sequence* current_sequ      # active sequence
    table_list* table_register  # list of all tables
    char_p_array* tmp_p_array   # temporary buffer for splits
    char_array* c_dum           # another temporary buffer
    var_list* variable_list     # globals
    el_list* element_list       # list of global elements


# Function declarations:
cdef extern from "madX/mad_api.h":
    sequence_list *madextern_get_sequence_list()

cdef extern from "madX/mad_core.h":
    void madx_start()
    void madx_finish()

cdef extern from "madX/mad_name.h":
    int name_list_pos(char*, name_list*)  # NOTE: C API uses "const char*"

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
    column_info table_get_column(char* table_name, char* column_name)
    char_p_array* table_get_header(char* table_name)
    int table_exists(char* table_name)

cdef extern from "madX/mad_var.h":
    # NOTE: C API uses "const char* name"
    variable* new_variable(char* name, double val, int val_type, int type, expression*, char* string)
    void add_to_var_list(variable*, var_list*, int flag)
    void set_variable(char* name, double* value)
    void set_stringvar(char* name, char* string)
    variable* find_variable(char* name, var_list*)
    double variable_value(variable*)


# I have no clue why, but for some reason, it is necessary to include
# 'madx.h' (or one of the file it includes). Otherwise, importing the Cython
# module will result in an ImportError: undefined symbol: pro_input
cdef extern from "madX/madx.h":
    pass
