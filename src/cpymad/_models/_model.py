import json, madx, os, sys

class model:
    ##
    # Initialize object
    def __init__(self,model,optics='',history=''):
        
        # initialize madx instance
        # history is an optional string,
        # which if defined will contain all madx
        # commands..
        self.madx=madx.madx(history)
        
        # loading the dictionary...
        try:
             import pkgutil
             _dict = pkgutil.get_data(__name__, model+'.json')
        except ImportError:
            import pkg_resources
            _dict = pkg_resources.resource_string(__name__,  model+'.json')
        self._dict=json.loads(_dict)
        
        self._db=None
        for db in self._dict['dbdir']:
            if os.path.isdir(db):
                self._db=db
        if not self._db:
            raise ValueError,"Could not find an available database path"
        
        
        self.madx.verbose(False)
        
        for cmd in self._dict['basecmds']:
            self.madx.command(cmd)
        self._call(self._dict['sequence'])
        
        self._optics=''
        self.set_optics(optics)

    def _call(self,f):
        self.madx.call(self._db+f)

    def set_optics(self,optics):
        if self._optics:
            print("INFO: Optics already initialized")
            return 0
        
        if not optics in self._dict['optics'].keys():
            print("INFO: Setting default optics")
            optics=self._dict['default']['optics']
        
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
        return self._dict['sequences']
    
    def list_optics(self):
        return self._dict['optics'].keys()
    
    def twiss(self,sequence=""):
        if sequence=="":
            sequence=self._dict['default']['sequence']
        return self.madx.twiss(sequence)
    