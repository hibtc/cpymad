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
    :param string optics: Name of optics to load, string or list of strings.
    :param string histfile: Name of file which will contain all Mad-X commands called.
    '''
    def __init__(self,model,sequence='',optics='',histfile='',recursive_history=False):

        if model not in cern.cpymad.modelList():
            raise ValueError("The model you asked for does not exist in the database")
        # name of model:
        self.model=model
        # path where the model file is located
        self._path = os.path.dirname(_get_filename(model))
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

        self._mprocess=_modelProcess(_child_pipe_send,_child_pipe_recv,model,histfile,recursive_history)

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


    def set_sequence(self,sequence='',madrange=''):
        '''
        Set a new active sequence...
        '''
        if not sequence:
            if not self._active['sequence']:
                self._active['sequence']=self._mdef['default-sequence']
            sequence=self._active['sequence']
        if sequence in self._mdef['sequences']:
            self._active['sequence']=sequence
            if madrange:
                self.set_range(madrange)
            else:
                self.set_range(self._mdef['sequences'][sequence]['default-range'])
        else:
            raise KeyError("You tried to activate a non-existing sequence")

    def set_range(self,madrange=''):
        '''
        Sets the active range to madrange. Must be defined in the
        currently active sequence...
        If madrange is empty, sets the range to default-range unless
        another range is already set.
        '''
        seqdict=self._mdef['sequences'][self._active['sequence']]
        if madrange:
            if madrange not in seqdict['ranges']:
                raise KeyError("%s is not a valid range name, available ranges: '%s'" % (madrange,"' '".join(seq['ranges'].keys())))
            self._active['range']=madrange
        else:
            if not self._active['range']:
                self._active['range']=seqdict['default-range']

    def _setup_initial(self,sequence,optics):
        '''
        Initial setup of the model
        '''
        for ifile in self._mdef['init-files']:
            if sys.flags.debug:
                print("Calling file: "+str(ifile))
            self._call(ifile)

        # initialize all sequences..
        for seq in self._mdef['sequences']:
            self._init_sequence(seq)
        # then we set the default one..
        self.set_sequence(sequence)
        if type(optics)==type(''):
            self.set_optic(optics)
        elif type(optics)==type([]):
            for o in optics:
                self.set_optic(o)
        # To keep track of whether or not certain things are already called..
        self._apercalled={}
        self._twisscalled={}
        for seq in self.get_sequences():
            self._apercalled[seq]=False
            self._twisscalled[seq]=False

    def _init_sequence(self,sequence):
        '''
        Initialize sequence
        '''
        bname=self._mdef['sequences'][sequence]['beam']
        bdict=self.get_beam(bname)
        self.set_beam(bdict)

    def get_beam(self,bname):
        '''
         Returns the beam definition in form
         of a dictionary.

         You can then change parameters in this dictionary
         as you see fit, and use set_beam() to activate that
         beam.
        '''
        return self._mdef['beams'][bname]

    def set_beam(self,beam_dict):
        '''
         Set the beam from a beam definition
         (dictionary)
        '''
        bcmd='beam'
        for k,v in beam_dict.items():
            bcmd+=','+k+'='+str(v)
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

    def _get_file_path(self,fdict):
        if 'location' in fdict:
            loc=fdict['location']
        else:
            loc='REPOSITORY' # this is default..
        if loc=='RESOURCE':
            fname = self._mdef["path-offsets"]['resource-offset']+'/'
        elif loc=='REPOSITORY':
            fname = self._mdef["path-offsets"]['repository-offset']+'/'

        if USE_COUCH:
            raise TypeError("Sorry, couch not implemented to use this feature")
        else:
            if loc=='RESOURCE':
                return self._path+'/resdata/'+fname
            elif loc=='REPOSITORY':
                if self._db:
                    return self._db+fname
                else:
                    return self._path+'/repdata/'+fname

    def _call(self,fdict):

        fpath=self._get_file_path(fdict)+fdict['path']
        self.call(fpath)


    def call(self,filepath):
        '''
         Call a file in Mad-X. Give either
         full file path or relative.
        '''
        if not os.path.isfile(filepath):
            raise ValueError("You tried to call a file that doesn't exist: "+filepath)

        if sys.flags.debug:
            print("Calling file: "+filepath)

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
        '''
         Returns a list of loaded sequences.
        '''
        return self._sendrecv('get_sequences')

    def list_optics(self):
        '''
         Returns a list of available optics
        '''
        return self._mdef['optics'].keys()

    def list_ranges(self,sequence=None):
        '''
         Returns a list of available ranges for the sequence.
         If sequence is not given, returns a dictionary structured as
         {sequence1:[range1,range2,...],sequence2:...}

         :param string sequence: sequence name.
        '''
        if sequence==None:
            ret={}
            for s in self.get_sequences():
                ret[s]=self._mdef['sequences'][s]['ranges'].keys()
            return ret

        return self._mdef['sequences'][sequence]['ranges'].keys()

    def list_beams(self):
        '''
         Returns a list of available beams
        '''
        return self._mdef['beams'].keys()

    def _get_twiss_initial(self,sequence='',madrange='',name=''):
        '''
        Returns the dictionary for the twiss initial conditions.
        If name is not defined, using default-twiss
        '''
        rangedict=self._get_range_dict(sequence=sequence,madrange=madrange)
        madrange=self._active['range']
        if name:
            if name not in rangedict['twiss-initial-conditions']:
                raise ValueError('twiss initial conditions with name '+name+' not found in range '+madrange)
            return rangedict['twiss-initial-conditions'][name]
        else:
            return rangedict['twiss-initial-conditions'][rangedict['default-twiss']]


    def twiss(self,
              sequence='',
              columns=['name','s','betx','bety','x','y','dx','dy','px','py','mux','muy','l','k1l','angle','k2l'],
              pattern=['full'],
              madrange='',
              fname='',
              retdict=False,
              use=True
              ):
        '''
         Run a TWISS on the model.

         Warning for ranges: Currently TWISS with initial conditions is NOT
         implemented!

         :param string sequence: Sequence, if empty, using active sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string madrange: Optional, give name of a range defined for the model.
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict).
         :param bool use: Call use before twiss.
        '''
        from cern.pymad.domain import TfsTable, TfsSummary

        # set sequence/range...
        if madrange:
            self.set_sequence(sequence,madrange)
        else:
            self.set_sequence(sequence)
        sequence=self._active['sequence']
        _madrange=self._active['range']

        if self._apercalled[sequence]:
            raise ValueError("BUG in Mad-X: Cannot call twiss after aperture..")

        seqdict=self._mdef['sequences'][sequence]
        rangedict=seqdict['ranges'][_madrange]

        args={'sequence':sequence,'columns':columns,'pattern':pattern,'fname':fname,'use':use}
        args['madrange']=[rangedict["madx-range"]["first"],rangedict["madx-range"]["last"]]
        args['twiss-init']=None
        if 'twiss-initial-conditions' in rangedict:
            args['twiss-init']={}
            for condition,value in self._get_twiss_initial(sequence,_madrange).items():
                if value:
                    args['twiss-init'][condition]=value
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
               retdict=False,
               use=True):
        '''
         Run a survey on the model.

         :param string sequence: Sequence, if empty, using active sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict).
         :param bool use: Call use before survey.
        '''
        self.set_sequence(sequence)
        sequence=self._active['sequence']

        this_range=''
        if madrange:
            rangedict=self._get_range_dict(sequence=sequence,madrange=madrange)
            this_range=rangedict['madx-range']

        args={'sequence':sequence,
              'columns':columns,
              'madrange':this_range,
              'fname':fname,
              'use':use}
        t,s=self._sendrecv(('survey',args))
        if retdict:
            return t,s
        from cern.pymad.domain import TfsTable, TfsSummary
        return TfsTable(t),TfsSummary(s)

    def aperture(self,
               sequence='',
               madrange='',
               columns='name,l,s,n1,aper_1,aper_2,aper_3,aper_4',
               fname='',
               retdict=False,
               use=False):
        '''
         Get the aperture from the model.

         :param string sequence: Sequence, if empty, using active sequence.
         :param string madrange: Range, if empty, the full sequence is chosen.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string fname: Optionally, give name of file for tfs table.
         :param bool retdict: Return dictionaries (default is an extended LookUpDict).
         :param bool use: Call use before aperture.
        '''
        from cern.pymad.domain import TfsTable, TfsSummary
        self.set_sequence(sequence)
        sequence=self._active['sequence']

        if not self._twisscalled[sequence]:
            self.twiss(sequence)
        # Calling "basic aperture files"
        if not self._apercalled[sequence]:
            for afile in seq['aperfiles']:
                print "Calling file",afile
                self._call(afile)
            self._apercalled[sequence]=True
        # getting offset file if any:
        # if no range was selected, we ignore offsets...
        offsets=''
        this_range=''
        if madrange:
            rangedict=self._get_range_dict(sequence=sequence,madrange=madrange)
            this_range=rangedict['madx-range']
            if 'aper-offset' in rangedict:
                offsets=rangedict['aper-offset']
                if USE_COUCH:
                    offsets_tmp='tmp_madx_offsets'
                    ocount=0
                    while os.path.isfile(offsets_tmp+str(ocount)+'.tfs'):
                        ocount+=1
                    offsets_tmp+=str(ocount)+'.tfs'
                    ftmp=file(offsets_tmp,'w')
                    ftmp.write(cern.cpymad._couch_server.get_file(self.model,offsets))
                    ftmp.close()
                    offsets=offsets_tmp
                else:
                    offsets=self._get_file_path(offsets)+offsets['path']

        args={'sequence':sequence,
              'madrange':this_range,
              'columns':columns,
              'offsets':offsets,
              'fname':fname,
              'use':use}
        t,s=self._sendrecv(('aperture',args))
        if USE_COUCH:
            os.remove(offsets)
        if retdict:
            return t,s
        return TfsTable(t),TfsSummary(s)

    def match(
            self,
            constraints,
            vary,
            method=['lmdif'],
            sequence = '',
            fname=''):
        """
        Perform a matching operation.
        """
        #----------------------------------------
        # from cern.pymad.domain import TfsTable, TfsSummary
        #----------------------------------------

        # set sequence/range...
        self.set_sequence(sequence)
        sequence=self._active['sequence']
        _madrange=self._active['range']

        seqdict=self._mdef['sequences'][sequence]
        rangedict=seqdict['ranges'][_madrange]

        args = {'sequence': sequence,
                'constraints': constraints,
                'vary': vary,
                'method': method,
                'fname': fname}
        args['madrange']=[rangedict["madx-range"]["first"],rangedict["madx-range"]["last"]]

        args['twiss-init']=None
        if 'twiss-initial-conditions' in rangedict:
            args['twiss-init']={}
            for condition,value in self._get_twiss_initial(sequence,_madrange).items():
                if value:
                    args['twiss-init'][condition]=value
        s=self._sendrecv(('match',args))

        #----------------------------------------
        # t,s=self._sendrecv(('match',args))
        # if retdict:
        #     return t,s
        # return TfsTable(t),TfsSummary(s)
        #----------------------------------------



    def _get_ranges(self,sequence):
        return self._mdef['sequences'][sequence]['ranges'].keys()

    def _get_range_dict(self,sequence='',madrange=''):
        '''
        Returns the range dictionary. If sequence/range isn't given,
        returns default for the model
        '''
        if sequence=='':
            sequence=self._active['sequence']
        elif sequence not in self._mdef['sequences']:
            raise ValueError("%s is not a valid sequence name, available sequences: '%s'" % (sequence,"' '".join(self._mdef['sequences'].keys())))

        seqdict=self._mdef['sequences'][sequence]
        if madrange:
            self.set_range(madrange)
        return seqdict['ranges'][self._active['range']]

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

    fname=_get_filename(modelname)
    _jdict=json.loads(_get_file_content(modelname,fname))
    ret_dict=_jdict[modelname]
    for extra_dict in ret_dict['extends']:
        for key in _jdict[extra_dict]:
            if key in ret_dict and not key=='real':
                if type(_jdict[extra_dict][key])==dict:
                    ret_dict[key].update(_jdict[extra_dict][key])
                elif type(_jdict[extra_dict][key])==list:
                    ret_dict[key].extend(_jdict[extra_dict][key])
                else:
                    raise TypeError('Could not extend %s model with %s' % (modelname,extra_dict))
            else:
                ret_dict[key]=_jdict[extra_dict][key]
        ret_dict.update(_jdict[extra_dict])

    return ret_dict

def _get_filename(modelname):
    from cern.cpymad import listModels
    file_list=listModels._get_mnames_files()[1]
    for f in file_list:
        if modelname in file_list[f]:
            return f
def _get_file_content(modelname,filename):
    if USE_COUCH:
        return cern.cpymad._couch_server.get_file(modelname,filename)

    try:
        with open(filename) as f:
            data = f.read()
    except FileNotFoundError:
        filename = os.path.join('_models',os.path.basename(filename))
        try:
            import pkgutil
            data = pkgutil.get_data(__name__, filename)
        except ImportError:
            import pkg_resources
            data = pkg_resources.resource_string(__name__, filename)
    return data


def save_model(model_def,filename):
    '''
    Saves the model definition defined by the dictionary model_def
    to file filename. The file is in json format. We do not check
    that the model is valid in any way.

    It is recommended that you rather use the modeldefs.model.save_model,
    once it is ready.
    '''
    if type(model_def)!=dict:
        raise TypeError('model_def must be a dictionary!')
    if type(filename)!=type(''):
        raise TypeError('filename must be a string!')
    file(filename,'w').write(json.dumps(model_def,indent=2))

class _modelProcess(multiprocessing.Process):
    def __init__(self,sender,receiver,model,history='',recursive_history=False):
        self.sender=sender
        self.receiver=receiver
        self.model=model
        self.history=history
        self.recursive_history=recursive_history
        multiprocessing.Process.__init__(self)

    def run(self):
        _madx=madx(histfile=self.history,recursive_history=self.recursive_history)
        _madx.verbose(False)
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
                                     fname=cmd[1]['fname'],
                                     twiss_init=cmd[1]['twiss-init'],
                                     use=cmd[1]['use'],
                                     retdict=True)
                    self.sender.send((t,s))
                elif cmd[0]=='survey':
                    t,s=_madx.survey(sequence=cmd[1]['sequence'],
                                     columns=cmd[1]['columns'],
                                     fname=cmd[1]['fname'],
                                     use=cmd[1]['use'],
                                     retdict=True)
                    self.sender.send((t,s))
                elif cmd[0]=='aperture':
                    t,s=_madx.aperture(sequence=cmd[1]['sequence'],
                                       madrange=cmd[1]['madrange'],
                                       columns=cmd[1]['columns'],
                                       offsets=cmd[1]['offsets'],
                                       fname=cmd[1]['fname'],
                                       use=cmd[1]['use'],
                                       retdict=True)
                    self.sender.send((t,s))
                elif cmd[0] == 'match':
                    _madx.match(
                            sequence=cmd[1]['sequence'],
                            constraints=cmd[1]['constraints'],
                            vary=cmd[1]['vary'],
                            method=cmd[1]['method'],
                            fname=cmd[1]['fname'],
                            twiss_init=cmd[1]['twiss-init'])
                    self.sender.send('done')
                    pass
                else:
                    raise ValueError("You sent a wrong command to subprocess: "+str(cmd))

