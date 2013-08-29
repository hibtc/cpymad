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

#cdef int NAME_L
#NAME_L=48 # defined by preprocessor in madx..

cdef extern from "madX/mad_def.h":
    #cdef NAME_L
    pass

cdef extern from "madX/madx.h":
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

cdef extern from "madX/mad_table.h":
    cdef struct table:
        char* name
        int num_cols, org_cols,dynamic,origin,curr
        char_p_array *header #,*node_nm
        int_array *col_out,*row_out
        name_list* columns    #names + types (in inform)
        char ***s_cols
        pass

      
cdef extern from "madX/madx.h":
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

    cdef struct column_info:
            void * data
            int length
            char datatype
            char datasize
