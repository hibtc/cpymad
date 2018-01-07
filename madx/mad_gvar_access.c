#include "madx.h"
#include "mad_gvar_access.h"
struct char_p_array*    gvar_get_tmp_p_array(void)      {return tmp_p_array;}
struct sequence*        gvar_get_current_sequ(void)     {return current_sequ;}
struct var_list*        gvar_get_variable_list(void)    {return variable_list;}
const char*             gvar_get_version_date(void)     {return version_date;}
struct char_array*      gvar_get_c_dum(void)            {return c_dum;}
struct el_list*         gvar_get_element_list(void)     {return element_list;}
const char*             gvar_get_version_name(void)     {return version_name;}
struct table_list*      gvar_get_table_register(void)   {return table_register;}