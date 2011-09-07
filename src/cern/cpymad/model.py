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
'''
.. module: cpymad.model

Cython implementation of the model api.
See also :py:class:`cern.pymad.model`

.. moduleauthor:: Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>

'''

import json, os, sys
from cern.madx import madx
import cern.cpymad
from cern.pymad import abc
import multiprocessing
import signal,atexit
from cern.pymad.globals import USE_COUCH

class model(abc.model.PyMadModel):
#class model():
    '''
    model class implementation. the model spawns a madx instance in a separate process.
    this has the advantage that you can run separate models which do not affect each other.
     
    :param string model: Name of model to load.
    :param string optics: Name of optics to load
    :param string histfile: Name of file which will contain all Mad-X commands called.
    '''
    def __init__(self,model,sequence='',optics='',histfile=''):
        
        if model not in cern.cpymad.modelList():
            raise ValueError("The model you asked for does not exist in the database")
        # name of model:
        self.model=model
        # loading the dictionary...
        self._mdef=_get_mdef(model)
        
        
        # Defining two pipes which are used for communicating...
        _child_pipe_recv,_parent_send=multiprocessing.Pipe(False)
        _parent_recv,_child_pipe_send=multiprocessing.Pipe(False)
        self._send=_parent_send.send
        self._recv=_parent_recv.recv
        self._db=None
        if not USE_COUCH:
            for d in self._mdef['dbdirs']:
                if os.path.isdir(d):
                    self._db=d
                    break
        #if self._db==None:
            #raise ValueError("It is not possible to find database directory for this model")
        
        self._mprocess=_modelProcess(_child_pipe_send,_child_pipe_recv,model,histfile)
        
        self._mprocess.start()
        
        atexit.register(self._mprocess.terminate)
        
        self._active={'optic':'','sequence':'','range':''}
        
        self._setup_initial(sequence,optics)
    
    # API stuff:
    @property
    def name(self):
        return self.model
    
    @property
    def mdef(self):
        return self._mdef.copy()
    
    def set_sequence(self,sequence):
        if sequence:
            if sequence in self._mdef['sequences']:
                self._active['sequence']=sequence
            else:
                print("WARNING: You tried to activate a non-existing sequence")
        elif not self._active['sequence']:
            self._active['sequence']=self._mdef['default-sequence']
        
    def _setup_initial(self,sequence,optics):
        
        for ifile in self._mdef['init-files']:
            self._call(ifile)
        
        # essentially just calls the beam command for all sequences..
        for seq in self._mdef['sequences']:
            self._set_sequence(seq)
        # then we set the default one..
        self.set_sequence(sequence)
            
        self.set_optic(optics)
        # To keep track of whether or not certain things are already called..
        self._apercalled={}
        self._twisscalled={}
        for seq in self.get_sequences():
            self._apercalled[seq]=False
            self._twisscalled[seq]=False
        
    def _set_sequence(self,sequence):
        sdict=self._mdef['sequences'][sequence]
        bdict=sdict['beam']
        bcmd='beam,'
        for k,v in bdict.items():
            bcmd+=k+'='+str(v)+','
        bcmd+='sequence='+sequence # remove last comma..
        if sys.flags.debug:
            print("Beam command: "+bcmd)
        self._cmd(bcmd)
        
    def __del__(self):
        try:
            self._send('delete_model')
            self._mprocess.join(5)
        except TypeError: pass
        except AttributeError: pass
    
    def __str__(self):
        return self.model
    
    def _call(self,fdict):
        if 'location' in fdict:
            loc=fdict['location']
        else:
            loc='REPOSITORY' # this is default..
        if loc=='RESOURCE':
            fname=self._mdef["path-offsets"]['resource-offset']+'/'+fdict['path']
        elif loc=='REPOSITORY':
            fname=self._mdef["path-offsets"]['repository-offset']+'/'+fdict['path']
        if USE_COUCH:
            cmd=cern.cpymad._couch_server.get_file(self.model,fname)
            for c in cmd.split('\n'): # I wonder if this might be needed for now?
                ret=self._sendrecv(('command',cmd))
            return ret
        else:
            if loc=='RESOURCE':
                fpath=os.path.dirname(__file__)+'/_models/resdata/'+fname
            elif loc=='REPOSITORY':
                if self._db:
                    fpath=self._db+fname
                else:
                    fpath=os.path.dirname(__file__)+'/_models/repdata/'+fname
        self.call(fpath)
    
    
    def call(self,filepath):
        '''
         Call a file in Mad-X. Give either
         full file path or relative to where script is ran
         from.
        '''
        if not os.path.isfile(filepath):
            raise ValueError("You tried to call a file that doesn't exist: "+filepath)
        
        if sys.flags.debug:
            print("Calling file: "+fpath)
        
        return self._sendrecv(('call',filepath))
    
    def has_sequence(self,sequence):
        '''
         Check if model has the sequence.
         
         :param string sequence: Sequence name to be checked.
        '''
        return sequence in self.get_sequences()
    
    def has_optics(self,optics):
        '''
         Check if model has the optics.
         
         :param string optics: Optics name to be checked.
        '''
        return optics in self._mdef['optics']
    
    def set_optic(self,optic):
        '''
         Set new optics.
         
         :param string optics: Optics name.
         
         :raises KeyError: In case you try to set an optics not available in model.
        '''
        
        if optic=='':
            optic=self._mdef['default-optic']
        if self._active['optic']==optic:
            print("INFO: Optics already initialized")
            return 0
        
        # optics dictionary..
        odict=self._mdef['optics'][optic]
        
        for strfile in odict['init-files']:
            self._call(strfile)
        
        # knobs dictionary.. we don't have that yet..
        #for f in odict['knobs']:
            #if odict['knobs'][f]:
                #self.set_knob(f,1.0)
            #else:
                #self.set_knob(f,0.0)
        
        self._active['optic']=optic
    
    def set_knob(self,knob,value):
        kdict=self._mdef['knobs']
        for e in kdict[knob]:
            val=str(kdict[knob][e]*value)
            self._cmd(e+"="+val)
    
    def get_sequences(self):
        return self._sendrecv('get_sequences')
    
    def list_optics(self):
        return self._mdef['optics'].keys()
    
    def twiss(self,
              sequence='',
              columns=['name','s','betx','bety','x','y','dx','dy','px','py','mux','muy','l','k1l','angle','k2l'],
              pattern=['full'],
              madrange='',
              fname='',
              retdict=False):
        '''
         Run a TWISS on the model.
         
         Warning for ranges: Currently TWISS with initial conditions is NOT
         implemented!
         
         :param string sequence: Sequence, if empty, using default sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string madrange: Optional, give name of a range defined for the model.
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict)
        '''
        from cern.pymad.domain import TfsTable, TfsSummary
        if sequence=='':
            sequence=self._mdef['default-sequence']
        args={'sequence':sequence,'columns':columns,'pattern':pattern,'madrange':madrange,'fname':fname}
        t,s=self._sendrecv(('twiss',args))
        # we say that when the "full" range has been selected, 
        # we can set this to true. Needed for e.g. aperture calls
        if not madrange:
            self._twisscalled[sequence]=True
        if retdict:
            return t,s
        return TfsTable(t),TfsSummary(s)
    
    def survey(self,
               sequence='',
               columns='name,l,s,angle,x,y,z,theta',
               madrange='',
               fname='',
               retdict=False):
        '''
         Run a survey on the model.
         
         :param string sequence: Sequence, if empty, using default sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict)
        '''
        from cern.pymad.domain import TfsTable, TfsSummary
        if sequence=='':
            sequence=self._mdef['default-sequence']
        args={'sequence':sequence,'columns':columns,'madrange':madrange,'fname':fname}
        t,s=self._sendrecv(('survey',args))
        if retdict:
            return t,s
        return TfsTable(t),TfsSummary(s)
    
    def aperture(self,
               sequence='',
               madrange='',
               columns='name,l,s,n1,aper_1,aper_2,aper_3,aper_4',
               fname='',
               retdict=False):
        '''
         Get the aperture from the model.
         
         :param string sequence: Sequence, if empty, using default sequence.
         :param string madrange: Range, if empty, the full sequence is chosen.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict)
        '''
        from cern.pymad.domain import TfsTable, TfsSummary
        if sequence=='':
            sequence=self._mdef['default-sequence']
        seq=self._mdef['sequences'][sequence]
        
        if not self._twisscalled[sequence]:
            self.twiss(sequence)
        # Calling "basic aperture files"
        if not self._apercalled[sequence]:
            for afile in seq['aperfiles']:
                self._call(afile)
            self._apercalled[sequence]=True
        # getting offset file if any:
        # if no range was selected, we ignore offsets...
        offsets=''
        if madrange:
            rangedict=self._get_range_dict(sequence,madrange)
            onum=seq['ranges'][madrange]['offsets']
            ran=seq['ranges'][madrange]['range']
            if onum or onum!=0:
                if USE_COUCH:
                    offsets='tmp_madx_offsets'
                    ocount=0
                    while os.path.isfile(offsets+str(ocount)+'.tfs'):
                        ocount+=1
                    offsets+=str(ocount)+'.tfs'
                    ftmp=file(offsets,'w')
                    ftmp.write(cern.cpymad._couch_server.get_file(self.model,seq['offsets'][onum]))
                    ftmp.close()
                else:
                    offsets=self._db+seq['offsets'][onum]
        
        args={'sequence':sequence,
              'madrange':ran,
              'columns':columns,
              'offsets':offsets,
              'fname':fname}
        t,s=self._sendrecv(('aperture',args))
        if USE_COUCH:
            os.remove(offsets)
        if retdict:
            return t,s
        return TfsTable(t),TfsSummary(s)
        
    def _get_ranges(self,sequence):
        return self._mdef['sequences'][sequence]['ranges'].keys()
        
    def _get_range_dict(self,sequence,madrange):
        seq=self._mdef['sequences'][sequence]
        if madrange not in seq['ranges']:
            raise ValueError("%s is not a valid range name, available ranges: '%s'" % (madrange,"' '".join(seq['ranges'].keys())))
        return seq['ranges'][madrange]
        
    def _cmd(self,command):
        self._send(('command',command))
        return self._recv()
    def _sendrecv(self,func):
        if sys.flags.debug:
            print("Sending function call "+str(func))
        self._send(func)
        return self._recv()
        
           
def _get_mdef(modelname):
    if USE_COUCH:
        return cern.cpymad._couch_server.get_model(modelname)
    fname=modelname+'.cpymad.json'
    _dict = _get_file_content(modelname,fname)
    return json.loads(_dict)[modelname]

def _get_file_content(modelname,filename):
    if USE_COUCH:
        return cern.cpymad._couch_server.get_file(modelname,filename)
    filename=os.path.join('_models',filename)
    try:
         import pkgutil
         stream = pkgutil.get_data(__name__, filename)
    except ImportError:
        import pkg_resources
        stream = pkg_resources.resource_string(__name__, filename)
    return stream
    
class _modelProcess(multiprocessing.Process):
    def __init__(self,sender,receiver,model,history=''):
        self.sender=sender
        self.receiver=receiver
        self.model=model
        self.history=history
        multiprocessing.Process.__init__(self) 
    
    def run(self):
        _madx=madx(histfile=self.history)
        _madx.verbose(False)
        _madx.append_model(self.model)
        if USE_COUCH:
            _couch_server=cern.cpymad._couch.couch.Server()
            _madx.command(_couch_server.get_file(self.model,'initscript'))

        def terminator(num, frame):
             sys.exit()
        signal.signal(signal.SIGTERM, terminator)

        while True:
            if self.receiver.poll(2):
                cmd=self.receiver.recv()
                if cmd=='delete_model':
                    self.sender.close()
                    self.receiver.close()
                    break
                elif cmd[0]=='call':
                    _madx.call(cmd[1])
                    self.sender.send('done')
                elif cmd=='get_sequences':
                    self.sender.send( _madx.get_sequences())
                elif cmd[0]=='command':
                    _madx.command(cmd[1])
                    self.sender.send('done')
                elif cmd[0]=='twiss':
                    t,s=_madx.twiss(sequence=cmd[1]['sequence'],
                                     columns=cmd[1]['columns'],
                                     pattern=cmd[1]['pattern'],
                                     fname=cmd[1]['fname']
                                     ,retdict=True)
                    self.sender.send((t,s))
                elif cmd[0]=='survey':
                    t,s=_madx.survey(sequence=cmd[1]['sequence'],
                                     columns=cmd[1]['columns'],
                                     fname=cmd[1]['fname']
                                     ,retdict=True)
                    self.sender.send((t,s))
                elif cmd[0]=='aperture':
                    t,s=_madx.aperture(sequence=cmd[1]['sequence'],
                                       madrange=cmd[1]['madrange'],
                                       columns=cmd[1]['columns'],
                                       offsets=cmd[1]['offsets'],
                                       fname=cmd[1]['fname'],
                                       retdict=True)
                    self.sender.send((t,s))
                else:
                    raise ValueError("You sent a wrong command to subprocess: "+str(cmd))

