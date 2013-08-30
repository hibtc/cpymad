
import json



class modeldef():
    '''
     Should eventually inherit the PyMad API...
    '''
    def __init__(self,modelfile,modelname):

        self._dict=json.load(file(modelfile,'r'))[modelname]
        self.name=modelname

        self._init_attr('sequences')
        self._init_attr('optics')

    def get_sequence(self,sequence):
        '''
         Returns the sequence definition as a dictionary..
        '''
        if not sequence in self.sequences.keys():
            raise ValueError("Sequence %s is not in the model" % sequence)
        return self.sequences[sequence].copy()

    def _init_attr(self,attr):
        setattr(self,attr,{})
        for a,adict in self._dict[attr].items():
            if attr=='sequences':
                getattr(self,attr)[a]=sequence(adict)
            elif attr=='optics':
                getattr(self,attr)[a]=optic(adict)
            else:
                raise ValueError("Do not know how to set attribute %s" % attr)

    def set_sequence(self, sequence_name, sequence_dict):
        self._dict['sequences'][sequence_name]=sequence_dict
        self.sequences[sequence_name]=sequence(sequence_dict)

    def save_model(self,filename):
        out_dict={self.name: self._dict}
        out_text=json.dumps(out_dict,indent=2)
        file(filename,'w').write(out_text)

    def copy(self):
        return self._dict.copy()

class sequence():
    def __init__(self,sequencedict):
        self._dict=sequencedict.copy()
        self.beam=beam(self._dict['beam'])
    def copy(self):
        return self._dict.copy()

class optic():
    def __init__(self,odict):
        self._dict=odict.copy()
        self.overlay=self._dict['overlay']
        self.init_files=self._dict['init-files'][:]

class beam():

    def __init__(self,beamdict):
        self._dict=beamdict.copy()
        for name,val in beamdict.items():
            setattr(self,name,val)

    def copy(self):
        return self._dict.copy()
