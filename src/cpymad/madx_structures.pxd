# This file holds the table struct as defined in madx.h
# 
# In the current state, this doesn't work very well,
# segfaults more often than not, and isn't helpful..
# 

#cdef int NAME_L
#NAME_L=48 # defined by preprocessor in madx..

cdef extern from "madxl.h":
    #cdef NAME_L
    pass
cdef extern from "madx.h":
    struct char_p_array:
        int flag,stamp
        char** p
        char[48] name
        int  max                     # max. array size
        int  curr                   # current occupation
        int* i
        pass
    struct int_array:
        char* name
        int curr
        int *i
        pass
    struct node:
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
    
    cdef struct table:
        char* name
        int num_cols, org_cols,dynamic,origin,curr
        char_p_array *header #,*node_nm
        int_array *col_out,*row_out
        name_list* columns    #names + types (in inform)
        char ***s_cols
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
    
