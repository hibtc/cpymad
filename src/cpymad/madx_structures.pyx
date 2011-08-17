# This file holds the table struct as defined in madx.h
# 
# In the current state, this doesn't work very well,
# segfaults more often than not, and isn't helpful..
# 
cdef int NAME_L
NAME_L=48 # defined by preprocessor in madx..
cdef extern from "madxl.h":
    #cdef NAME_L
    pass
cdef extern from "madx.h":
    #char[NAME_L] name
    #int  max,                     # max. array size
    #curr,                    # current occupation
    #flag;                    # ancillary flag
    #int stamp;
    #char** p;
    struct char_p_array:
        #int curr,flag,stamp
        #char** p
        int stamp
        char[48] name
        #int  max,                     # max. array size
        #     curr;                    # current occupation
        #int* i;
        pass
    struct int_array:
        pass
        #char* name
        #int curr
        #int *i
    struct node:
        pass
    #char name[NAME_L],
    #type[NAME_L];            # like "twiss", "survey" etc.
    #int  max,                     # max. # rows
    #curr,                    # current # rows
    #num_cols,                # total # columns - fixed
    #org_cols,                # original # columns from definition
    #dynamic,                 # if != 0, values taken from current row
    #origin;                  # 0 if created in job, 1 if read
    #struct char_p_array* header;  # extra lines for file header
    #struct int_array* col_out;    # column no.s to be written (in this order)
    #struct int_array* row_out;    # flag for row: 1 write, 0 don't
    #struct char_p_array* node_nm; # names of nodes at each row
    #struct char_p_array** l_head; # extra lines to be put in front of a line
    #struct node** p_nodes;        # pointers to nodes at each row
    #char*** s_cols;               # string columns
    #double** d_cols;              # double precision columns
    #int stamp;
    #struct name_list* columns;    # names + types (in inform):
    #                                   1 double, 3 string
    #struct sequence* org_sequ;    # pointer to sequence it refers to
    cdef struct table:
        char* name
        int num_cols #,org_cols,dynamic,origin
        #char_p_array *header,*node_nm
        #int_array *col_out,*row_out
        pass
    
    struct name_list:
          char[48]  name           
          int  max                      # max. pointer array size
          int  curr                     # current occupation
          #int* index                    # index for alphabetic access
          #int* inform                   # array parallel to names with integer
          #int stamp
          #char** names;                 # element names for sort
          pass
      
    # to be able to read sequence information..
    struct sequence:
        char[48] name
        table* tw_table       #pointer to latest twiss table created
        pass
    # list of sequences..
    struct sequence_list:
          sequence_list *list       # index list of names
          sequence **sequs      # sequence pointer list
          int curr
          pass
    