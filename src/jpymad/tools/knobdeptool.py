import numpy as np

class KnobDepTool():
    '''
    This class provides some simple functions to calc parameter
    - dependencies of the model 
    '''
    def __init__(self, model, knob):
        self._model = model
        self._knob = knob
        
    def calc(self, madxvars=['x', 'mux'], \
        paramrange=np.linspace(-0.002, 0.002, 21), \
        elementpatterns=['.*']):
        '''
        this method calculates the dependency of the given variable 
        on the given knob
        '''
                
        oldvalue = self._knob.getOffset()
        
        columns = list(madxvars)
        columns.append('name')
        columns.append('s')
        
        data = None
        
        for paramvalue in paramrange:
            self._knob.setOffset(float(paramvalue))
            twresult = self._model.twiss(columns, elementpatterns)
            
            if data == None:
                data = Data()
                data.name = np.array(twresult['name'])
                data.s = np.array(twresult['s'])
                data.paramrange = paramrange
                for madxvar in madxvars:
                    #print madxvar + ": len=" + len(twresult[madxvar])
                    setattr(data, madxvar, np.array([twresult[madxvar]]))
            else:
                for madxvar in madxvars:
                    #print madxvar + ": len=" + len(twresult[madxvar])
                    setattr(data, madxvar, np.append(getattr(data, madxvar), \
                                     np.array([twresult[madxvar]]), axis=0))
                    
        self._knob.setOffset(oldvalue)
        
        return data
            
class Data(object):
    pass
