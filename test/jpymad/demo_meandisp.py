'''
Created on Sep 1, 2009

calculates the mean dispersion at the MBIs

@author: kfuchsbe
'''

from pymad import am, connect, set_am
import numpy as np
from pymad.tools_models import create_model

def ensure_model():
    '''
    ensures, that the desired model is created and active
    '''
    connect()
    active_model = am()
    if active_model is None:
        new_model = create_model("TI8")
        set_am(new_model)
    
def calc_mean_dispersion(elementpatterns=['MBI[T\.].*']):
    """
    caluculates the mean dispersion at the elements, which match the given regex
    """
    connect()
    
    model = am()
    tw = model.twiss(['name','dx'], elementpatterns)
    print 'elements:'
    print '------'
    print tw['name']
    print 'mean of dispersion on elements=' + str(np.average(tw['dx']))


if __name__ == "__main__":
    ensure_model()
    calc_mean_dispersion()
