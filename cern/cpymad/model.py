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
"""
Cython implementation of the model api.
"""

from __future__ import absolute_import

import logging
import os
import sys
import yaml

from ..resource.package import PackageResource
from .model_locator import MergedModelLocator
from .madx import Madx


__all__ = [
    'Model',
    'Factory',
    'default_factory',
    'get_model_names',
    'load_model',
    'save_model',
]


class Model(object):
    '''
    Model class implementation. the model spawns a madx instance in a separate process.
    this has the advantage that you can run separate models which do not affect each other.
    '''

    def __init__(self, mdata, madx, logger, sequence, optics):
        """
        Initialize a Model object.

        Users should use the load_model function instead.

        :param ModelData mdata: model data as acquired through a ModelLocator
        :param str sequence: Name of the default sequence to use
        :param str optics: Name of optics to load, string or list of strings.
        :param Madx madx: MAD-X instance to use
        :param logging.Logger logger:
        """
        self.madx = madx
        self._log = logger
        self.mdata = mdata
        self._mdef = mdata.model
        self._active={'optic': None, 'sequence': None, 'range': None}
        self._setup_initial(sequence,optics)

    # API stuff:
    @property
    def name(self):
        return self.mdata.name

    @property
    def mdef(self):
        return self._mdef.copy()


    def set_sequence(self,sequence=None, range=None):
        '''
        Set a new active sequence...
        '''
        if not sequence:
            if not self._active['sequence']:
                self._active['sequence']=self._mdef['default-sequence']
            sequence=self._active['sequence']
        if sequence in self._mdef['sequences']:
            self._active['sequence']=sequence
            if range:
                self.set_range(range)
            else:
                self.set_range(self._mdef['sequences'][sequence]['default-range'])
        else:
            raise KeyError("You tried to activate a non-existing sequence")

    def set_range(self, range=None):
        '''
        Sets the active range to range. Must be defined in the
        currently active sequence...
        If range is empty, sets the range to default-range unless
        another range is already set.
        '''
        seqdict=self._mdef['sequences'][self._active['sequence']]
        if range:
            if range not in seqdict['ranges']:
                raise KeyError("%s is not a valid range name, available ranges: '%s'" % (range,"' '".join(seq['ranges'].keys())))
            self._active['range']=range
        else:
            if not self._active['range']:
                self._active['range']=seqdict['default-range']

    def _setup_initial(self,sequence,optics):
        '''
        Initial setup of the model
        '''
        for ifile in self._mdef['init-files']:
            self._call(ifile)

        # initialize all sequences..
        for seq in self._mdef['sequences']:
            self._init_sequence(seq)
        # then we set the default one..
        self.set_sequence(sequence)
        if isinstance(optics, list):
            for o in optics:
                self.set_optic(o)
        else: # str/unicode/None
            self.set_optic(optics)
        # To keep track of whether or not certain things are already called..
        self._apercalled={}
        self._twisscalled={}
        for seq in self.get_sequences():
            self._apercalled[seq.name]=False
            self._twisscalled[seq.name]=False

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
        self.madx.command.beam(**beam_dict)

    def __str__(self):
        return self.name

    def _call(self,fdict):
        with self.mdata.get_by_dict(fdict).filename() as fpath:
            self.call(fpath)

    def call(self,filepath):
        '''
         Call a file in Mad-X. Give either
         full file path or relative.
        '''
        if not os.path.isfile(filepath):
            raise ValueError("You tried to call a file that doesn't exist: "+filepath)

        self._log.debug("Calling file: %s", filepath)
        return self.madx.call(filepath)

    def has_sequence(self,sequence):
        """
        Check if model has the sequence.

        :param string sequence: Sequence name to be checked.
        """
        return sequence in self.get_sequence_names()

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

        if not optic:
            optic=self._mdef['default-optic']
        if self._active['optic'] == optic:
            self._log.info("Optics already initialized: %s", optic)
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
        kdict = self._mdef['knobs']
        for e in kdict[knob]:
            val = kdict[knob][e] * value
            self.madx.command(**{e: val})

    def get_sequences(self):
        """
        Return iterable of sequences defined in the model.
        """
        return (self.madx.get_sequence(name)
                for name in self.get_sequence_names())

    def get_sequence_names(self):
        """
        Return iterable of all sequences defined in the model.
        """
        return self._mdef['sequences'].keys()

    def list_optics(self):
        '''
         Returns an iterable of available optics
        '''
        return self._mdef['optics'].keys()

    def list_ranges(self,sequence=None):
        '''
         Returns a list of available ranges for the sequence.
         If sequence is not given, returns a dictionary structured as
         {sequence1:[range1,range2,...],sequence2:...}

         :param string sequence: sequence name.
        '''
        if sequence is None:
            ret={}
            for s in self.get_sequences():
                ret[s.name]=list(self._mdef['sequences'][s]['ranges'].keys())
            return ret

        return list(self._mdef['sequences'][sequence]['ranges'].keys())

    def list_beams(self):
        '''
         Returns an iterable of available beams
        '''
        return self._mdef['beams'].keys()

    def _get_twiss_initial(self, sequence=None, range=None, name=None):
        '''
        Returns the dictionary for the twiss initial conditions.
        If name is not defined, using default-twiss
        '''
        rangedict=self._get_range_dict(sequence=sequence,range=range)
        range=self._active['range']
        if name:
            if name not in rangedict['twiss-initial-conditions']:
                raise ValueError('twiss initial conditions with name '+name+' not found in range '+range)
            return rangedict['twiss-initial-conditions'][name]
        else:
            return rangedict['twiss-initial-conditions'][rangedict['default-twiss']]


    def twiss(self,
              sequence=None,
              columns=['name','s','betx','bety','x','y','dx','dy','px','py','mux','muy','l','k1l','angle','k2l'],
              pattern=['full'],
              range=None,
              use=True,
              **kwargs):
        '''
         Run a TWISS on the model.

         Warning for ranges: Currently TWISS with initial conditions is NOT
         implemented!

         :param string sequence: Sequence, if empty, using active sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param string range: Optional, give name of a range defined for the model.
         :param bool use: Call use before twiss.
         :param kwargs: further keyword arguments for the MAD-X command
        '''
        # set sequence/range...
        if range:
            self.set_sequence(sequence,range)
        else:
            self.set_sequence(sequence)
        sequence=self._active['sequence']
        _range=self._active['range']

        if self._apercalled.get(sequence):
            raise ValueError("BUG in Mad-X: Cannot call twiss after aperture..")

        seqdict=self._mdef['sequences'][sequence]
        rangedict=seqdict['ranges'][_range]

        if 'twiss-initial-conditions' in rangedict:
            # this looks like a bug check to me (0 evaluates to False):
            twiss_init = dict(
                (key, val)
                for key, val in self._get_twiss_initial(sequence,_range).items()
                if val)
        else:
            twiss_init = None

        res = self.madx.twiss(
            sequence=sequence,
            pattern=pattern,
            columns=columns,
            range=[rangedict["madx-range"]["first"],rangedict["madx-range"]["last"]],
            twiss_init=twiss_init,
            use=use,
            **kwargs)
        # we say that when the "full" range has been selected,
        # we can set this to true. Needed for e.g. aperture calls
        if not range:
            self._twisscalled[sequence]=True
        return res

    def survey(self,
               sequence=None,
               columns='name,l,s,angle,x,y,z,theta',
               range=None,
               use=True,
               **kwargs):
        '''
         Run a survey on the model.

         :param string sequence: Sequence, if empty, using active sequence.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param bool use: Call use before survey.
         :param kwargs: further keyword arguments for the MAD-X command
        '''
        self.set_sequence(sequence)
        sequence=self._active['sequence']

        this_range=None
        if range:
            rangedict=self._get_range_dict(sequence=sequence,range=range)
            this_range=rangedict['madx-range']

        return self.madx.survey(
            sequence=sequence,
            columns=columns,
            range=this_range,
            use=use,
            **kwargs)

    def aperture(self,
               sequence=None,
               range=None,
               columns='name,l,s,n1,aper_1,aper_2,aper_3,aper_4',
               use=False,
               **kwargs):
        '''
         Get the aperture from the model.

         :param string sequence: Sequence, if empty, using active sequence.
         :param string range: Range, if empty, the full sequence is chosen.
         :param string columns: Columns in the twiss table, can also be list of strings
         :param bool use: Call use before aperture.
         :param kwargs: further keyword arguments for the MAD-X command
        '''
        self.set_sequence(sequence)
        sequence=self._active['sequence']

        if not self._twisscalled.get(sequence):
            self.twiss(sequence)
        # Calling "basic aperture files"
        if not self._apercalled[sequence]:
            for afile in self._mdef['sequences'][sequence]['aperfiles']:
                self._call(afile)
            self._apercalled[sequence]=True
        # getting offset file if any:
        # if no range was selected, we ignore offsets...
        offsets=None
        this_range=None
        if range:
            rangedict=self._get_range_dict(sequence=sequence,range=range)
            this_range=rangedict['madx-range']
            if 'aper-offset' in rangedict:
                offsets = self.mdata.get_by_dict(rangedict['aper-offset']).filename()

        args={'sequence': sequence,
              'range': this_range,
              'columns': columns,
              'use': use}
        args.update(kwargs)

        if offsets:
            with offsets as offsets_filename:
                return self.madx.aperture(offsets=offsets_filename, **args)
        else:
            return self.madx.aperture(**args)


    def match(
            self,
            constraints,
            vary,
            weight=None,
            method=['lmdif'],
            sequence=None,
            knobfile=None):
        """
        Perform a matching operation.

        See :func:`cern.madx.match` for a description of the parameters.
        """
        # set sequence/range...
        self.set_sequence(sequence)
        sequence=self._active['sequence']
        _range=self._active['range']

        seqdict=self._mdef['sequences'][sequence]
        rangedict=seqdict['ranges'][_range]

        def is_match_param(v):
            return v.lower() in ['rmatrix', 'chrom', 'beta0', 'deltap',
                    'betx','alfx','mux','x','px','dx','dpx',
                    'bety','alfy','muy','y','py','dy','dpy' ]

        if 'twiss-initial-conditions' in rangedict:
            twiss_init = dict(
                (key, val)
                for key, val in self._get_twiss_initial(sequence,_range).items()
                if is_match_param(key))
        else:
            twiss_init = None

        self.madx.match(
            sequence=sequence,
            constraints=constraints,
            vary=vary,
            weight=weight,
            method=method,
            knobfile=knobfile,
            twiss_init=twiss_init)
        return self.twiss(sequence=sequence)

    def _get_ranges(self,sequence):
        return self._mdef['sequences'][sequence]['ranges'].keys()

    def _get_range_dict(self, sequence=None, range=None):
        '''
        Returns the range dictionary. If sequence/range isn't given,
        returns default for the model
        '''
        if not sequence:
            sequence=self._active['sequence']
        elif sequence not in self._mdef['sequences']:
            raise ValueError("%s is not a valid sequence name, available sequences: '%s'" % (sequence,"' '".join(self._mdef['sequences'].keys())))

        seqdict=self._mdef['sequences'][sequence]
        if range:
            self.set_range(range)
        return seqdict['ranges'][self._active['range']]


class Factory(object):

    """Model instance factory."""

    def __init__(self, model_locator, model_cls=None):
        """Create Model factory using a specified ModelLocator."""
        self._model_locator = model_locator
        self._model_cls = model_cls or Model

    def get_model_names(self):
        """Get iterable over all model names."""
        return self._model_locator.list_models()

    def _find(self, name):
        """Find model definition, return ModelData."""
        return self._model_locator.get_model(name)

    def _create(self, mdata, sequence, optics, madx, histfile, logger):
        """
        Create Model instance based on ModelData.

        Parameters as in load_model (except for mdata).
        """
        if madx is None:
            madx = Madx(histfile)
            madx.verbose(False)
        elif histfile is not None:
            raise ValueError("'histfile' cannot be used with 'madx'")
        if logger is None:
            logger = logging.getLogger(__name__)
        return self._model_cls(mdata,
                               sequence=sequence,
                               optics=optics,
                               madx=madx,
                               logger=logger)

    def load_model(self,
                   name,
                   # *,
                   # These should be passed as keyword-only parameters:,
                   sequence=None,
                   optics=None,
                   madx=None,
                   histfile=None,
                   logger=None):
        """
        Find model definition by name and create Model instance.

        :param str name: model name
        :param str sequence: Name of the initial sequence to use
        :param str optics: Name of optics to load, string or list of strings.
        :param Madx madx: MAD-X instance to use
        :param str histfile: history file name; use only if madx is None!
        :param logging.Logger logger:
        """
        return self._create(self._find(name),
                            sequence=sequence,
                            optics=optics,
                            madx=madx,
                            histfile=histfile,
                            logger=logger)


_default_resources = PackageResource(__package__, '_models')
_default_locator = MergedModelLocator(_default_resources)
default_factory = Factory(_default_locator)

get_model_names = default_factory.get_model_names
load_model = default_factory.load_model


def save_model(model_def,filename):
    '''
    Saves the model definition defined by the dictionary model_def
    to file filename. The file is in YAML format (or JSON). We do not check
    that the model is valid in any way.

    It is recommended that you rather use the modeldefs.model.save_model,
    once it is ready.
    '''
    yaml.safe_dump(model_def,stream=open(filename,'w'))

