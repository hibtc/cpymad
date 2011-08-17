'''
contains convenience funtions to initialize the pymad envirenment and to clean up

Created on 16 Aug 2011

@author: kfuchsbe
'''

from pymad.globals import PyMadGlobals

class PyMadNotInititalizedError(Exception):
    pass

def get_pms():
    ''' Returns the pymad-service singleton. If pymad is not initialized this raises an error '''
    if not is_initialized():
        raise(PyMadNotInititalizedError())
    
    return PyMadGlobals.PYMAD_SERVICE

def is_initialized():
    ''' returns true, if a pymad service is initialized, false otherwise '''
    return (not PyMadGlobals.PYMAD_SERVICE is None)

def init(mode='cpymad', **kwargs):
    ''' Initializes the environment for using pymad in one of the possible modes. The first argument determines
    the mode (allowed values are 'cpymad' and 'jpymad'. If already a pymad service is running, then this function
    prints a warning and does nothing else. 
    
    Arguments:
    :param mode: Can be either 'cpymad' or 'pymad'. This option determines which pymad-service will be initialized. The default is 'cpymad'.
    :param kwargs: All the rest of the arguments are directly passed to the constructor of the created PyMadService
    :returns: the newly created pymad service if one was created.
     
    '''
    
    if is_initialized():
        print("Already a pymad service running. Doing nothing.")
        return None
    
    if mode is 'cpymad':
        from cpymad.service import CpymadService 
        pms = CpymadService(**kwargs)
    elif mode is 'jpymad':
        from jpymad.service import JPyMadService
        pms = JPyMadService(**kwargs)
    else:
        raise ValueError("Unknown mode '" + mode + "'! Use one of 'cpymad' or 'jpymad'!")
    
    PyMadGlobals.PYMAD_SERVICE = pms
    return pms

def cleanup():
    ''' Cleans up the pymad service and sets the global variable back to None again. '''
    
    if not is_initialized():
        print "pymad is not initialized. Doing nothing."
        return
    
    pms = PyMadGlobals.PYMAD_SERVICE
    pms.cleanup()
    PyMadGlobals.PYMAD_SERVICE = None
    
    
    
