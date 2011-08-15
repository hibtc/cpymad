'''
Created on Sep 1, 2009

calculates the mean dispersion at the MBIs

@author: kfuchsbe
'''

import numpy as np
from jpymad.service import JPyMadService

pms = None

def ensure_model(pms):
    '''
    ensures, that the desired model is created and active
    '''
    active_model = pms.am()
    if active_model is None:
        new_model = pms.create_model("TI8")
        pms.set_am(new_model)
    
def calc_mean_dispersion(model,elementpatterns=['MBI[T\.].*']):
    """
    calculates the mean dispersion at the elements, which match the given regex
    """
    tw = model.twiss(['name','dx'], elementpatterns)
    print 'elements:'
    print '------'
    print tw['name']
    print 'mean of dispersion on elements=' + str(np.average(tw['dx']))


if __name__ == "__main__":
    pms = JPyMadService()
    ensure_model(pms)
    model = pms.am()
    calc_mean_dispersion(model)
