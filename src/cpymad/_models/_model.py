import json, os, sys
from madx import madx


class model:
    ##
    # Initialize object
    def __init__(self,model,optics='',history=''):
        # initialize madx instance
        # history is an optional string,
        # which if defined will contain all madx
        # commands..
        self.madx=madx(history)
        
        already_loaded=False
        for m in self.madx.list_of_models():
            if model==m.model:
                _deepcopy(m,self)
                already_loaded=True
        
        if not already_loaded:
            # check that the model does not conflict with existing models..
            _check_compatible(model)
            # name of model:
            self.model=model
            # loading the dictionary...
            self._dict=json.loads(_get_data(model+'.json'))
            
            self._db=None
            for db in self._dict['dbdir']:
                if os.path.isdir(db):
                    self._db=db
            if not self._db:
                raise ValueError,"Could not find an available database path"
            
            
            self.madx.verbose(False)
            self.madx.command(_get_data(self._dict['header']))
            
            self._call(self._dict['sequence'])
            
            self._optics=''
            self.madx.append_model(self)
        
        self.set_optics(optics)
    
    # unsure why it doesn't recognize that madx is out of scope by itself..
    def __del__(self):
        del self.madx
    
    def _call(self,f):
        self.madx.call(self._db+f)
    
    def has_sequence(self,sequence):
        return sequence in self.madx.get_sequence_list()
    
    def has_optics(self,optics):
        return optics in self._dict['optics']
        
    def set_optics(self,optics):
        if optics=='':
            optics=self._dict['default']['optics']
            
        if self._optics==optics:
            print("INFO: Optics already initialized")
            return 0
        
        # optics dictionary..
        odict=self._dict['optics'][optics]
        # flags dictionary..
        fdict=self._dict['flags']
        bdict=self._dict['beams']
        
        for strfile in odict['strengths']:
            self._call(strfile)
        
        for f in odict['flags']:
            val=odict['flags'][f]
            for e in fdict[f]:
                self.madx.command(e+":="+val)
        
        for b in odict['beams']:
            self.madx.command(bdict[b])
        self._optics=optics
    
    def list_sequences(self):
        return self.madx.get_sequence_list()
    
    def list_optics(self):
        return self._dict['optics'].keys()
    
    def twiss(self,sequence=""):
        if sequence=="":
            sequence=self._dict['default']['sequence']
        return self.madx.twiss(sequence)


def _deepcopy(origin,new):
    new.madx=origin.madx
    new.model=origin.model
    new._dict=origin._dict
    new._db=origin._db
    new._optics=origin._optics
  
def _get_data(filename):
    try:
         import pkgutil
         _dict = pkgutil.get_data(__name__, filename)
    except ImportError:
        import pkg_resources
        _dict = pkg_resources.resource_string(__name__,  filename)
    return _dict  

def _check_compatible(model):
    m=madx()
    d=json.loads(_get_data(model+'.json'))
    if len(m.list_of_models())>0:
        # we always throw error until we know this actually can work..
        raise ValueError("Two models cannot be loaded at once at this moment")