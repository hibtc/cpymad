"""
Cython implementation of the model api.
"""

from __future__ import absolute_import

import collections
import logging
import os

from . import madx
from . import util
from ..resource.file import FileResource


__all__ = [
    'Model',
    'Factory',
    'Locator',
]


class Model(object):

    """
    A model is a complete description of an accelerator machine.

    This class is used to bundle all metadata related to an accelerator and
    all its configurations. It takes care of loading the proper MAD-X files
    when needed.


    Private members
    ~~~~~~~~~~~~~~~

    Underlying resources and handlers:

    :ivar str name: resource access
    :ivar str mdef: model definition data
    :ivar str _repo: resource access
    :ivar Madx _madx: handle to the MAD-X library

    Names of active optic/sequence/range:

    :ivar str _active_optic:
    :ivar str _active_sequence:
    :ivar str _active_dict:

    Cached references to active optic/sequence/range definition:

    :ivar dict _active_optic_def:
    :ivar dict _active_sequence_def:
    :ivar dict _active_range_def:

    Keep track whether TWISS/APERTURE commands have been called:

    :ivar dict _aperture_called:
    :ivar dict _twiss_called:
    """

    def __init__(self, name, mdef, repo, madx, logger):
        """
        Initialize a Model object.

        Users should use the load_model function instead.

        :param Madx madx: MAD-X instance to use
        :param logging.Logger logger:
        """
        # init instance variables
        self._name = name
        self._mdef = mdef
        self._repo = repo
        self._madx = madx
        self._log = logger
        self._active_sequence = None
        self._active_sequence_def = None
        self._active_range = None
        self._active_range_def = None
        self._active_optic = None
        self._active_optic_def = None
        self._aperture_called = {}
        self._twiss_called = {}
        # call common initialization files
        self._call(*self._mdef['init-files'])
        # initialize all sequence beams, since many MAD-X commands need these
        # to be set:
        for seq in self._mdef['sequences']:
            bname = self._mdef['sequences'][seq]['beam']
            bdict = self.get_beam(bname)
            self.set_beam(bdict)

    # basic accessors
    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, self.name)

    @property
    def name(self):
        """Return model name."""
        return self._name

    @property
    def mdef(self):
        """Return model definition dictionary (can be serialized as YAML)."""
        return self._mdef

    @property
    def madx(self):
        """Return underlying :class:`Madx` instance."""
        return self._madx

    # Manage the current sequence/range selection:
    def set_optic(self, optic=None):
        """
        Select optic in the model definition.

        :param str optic: optic name.
        :raises KeyError: if the optic is not defined in the model
        """
        _optic = (optic or
                  self.get_default_optic())
        if _optic == self._active_optic:
            return
        self._call(*self._get_optic_def(_optic).get('init-files', ()))
        self._active_optic = optic

    def set_sequence(self, sequence=None, range=None):
        """
        Select active sequence and range by name.

        :param str sequence: sequence name in the model definition
        :param str range: range name in the model definition
        :returns: activated sequence name
        :rtype: str
        :raises KeyError: if sequence or range is not defined

        If left empty, sequence and range default to the current selection
        or default values in the model definition.

        Calling this function also causes a USE command in the MAD-X process
        if the active sequence is changed.
        """
        _sequence = (sequence or
                     self._active_sequence or
                     self.get_default_sequence())
        if _sequence != self._active_sequence:
            self._active_sequence = _sequence
            # raises:
            self._active_sequence_def = self._get_sequence_def(_sequence)
            # range must be reset when changing the sequence:
            self._active_range = None
            self._active_range_def = None
            # USE the sequence to make it the active sequence in MAD-X:
            self._madx.use(_sequence)
        self.set_range(range)
        return _sequence

    def set_range(self, range=None):
        """
        Select active range in the current sequence.

        :param str range: range name in the model definition
        :returns: activated range name
        :rtype: str
        :raises KeyError: if the range is not defined

        If left empty, range defaults to the current selection or default
        value of the current sequence.
        """
        _range = (range or
                  self._active_range or
                  self.get_default_range())
        # raises KeyError:
        self._active_range_def = self._active_sequence_def['ranges'][_range]
        self._active_range = range
        return range

    def set_beam(self, beam):
        """
        Select beam from the model definition.

        :param str name: beam name
        :raises KeyError: if the beam is not defined
        """
        self._madx.beam(**self.get_beam(beam))


    # accessors for model definition

    def get_default_optic(self):
        return self._mdef['default-optic']

    def get_default_sequence(self):
        return self._mdef['default-sequence']

    def get_default_range(self, sequence=None):
        """
        """
        return self._get_sequence_def(sequence)['default-range']

    def get_sequence(self):
        """Get name of the active sequence."""
        return self._active_sequence

    def has_sequence(self, sequence):
        """
        Check if model has the sequence.

        :param str sequence: Sequence name to be checked.
        """
        return sequence in self.get_sequences()

    def has_optics(self,optics):
        """
        Check if model has the optics.

        :param str optics: Optics name to be checked.
        """
        return optics in self._mdef['optics']


    def get_beams(self):
        """
        Return an iterable over all available beams in the model.

        :returns: iterable over all beam names
        :rtype: dict
        """
        return self._mdef['beams']

    def get_beam(self, beam):
        """
        Return the beam definition from the model.

        :param str beam: beam name
        :returns: beam definition
        :rtype: dict
        """
        return self._mdef['beams'][beam]

    def get_optics(self):
        """
        Return an iterable over all optics in the model.

        :returns: iterable over optic names
        :rtype: dict
        """
        return self._mdef['optics']

    def get_sequences(self):
        """
        Return an iterable over all sequences in the model.

        :returns: iterable over sequence names
        :rtype: dict
        """
        return (self.madx.get_sequence(name)
                for name in self.get_sequence_names())

    def get_sequence_names(self):
        """
        Return iterable of all sequences defined in the model.
        """
        return self._mdef['sequences'].keys()


    def get_ranges(self, sequence=None):
        """
        Return an iterable over all range definitions in the sequence.

        :param str sequence: sequence name, if empty take active sequence
        :returns: iterable over range names
        :rtype: dict
        :raises KeyError: if the sequence is not defined
        """
        return self._get_sequence_def(sequence)['ranges']

    # convenience accessors

    def get_active_optic(self):
        """
        Get the active optic.
        """
        return self._active_optic

    def set_active_optic(self, name):
        """
        """
        self.set_optic(name)

    def get_active_sequence(self):
        """
        Return name of the active sequence.

        :rtype: str
        """
        return self._active_sequence

    def set_active_sequence(self, name):
        """
        Set name of the active sequence.

        :param str name: sequence name
        """
        self.set_sequence(name)

    active_optic = property(get_active_optic, set_active_optic)
    active_sequence = property(get_active_sequence, set_active_sequence)

    default_optic = property(get_default_optic)
    default_sequence = property(get_default_sequence)
    default_range = property(get_default_range)

    # OTHER

    def call(self, filepath):
        """
        Call a file in Mad-X. Give either full file path or relative.
        """
        if not os.path.isfile(filepath):
            raise ValueError("You tried to call a file that doesn't exist: "+filepath)

        self._log.debug("Calling file: %s", filepath)
        return self.madx.call(filepath)

    def set_knob(self, knob, value):
        kdict = self._mdef['knobs']
        for e in kdict[knob]:
            val = kdict[knob][e] * value
            self.madx.command(**{e: val})

    def twiss(self, sequence=None, range=None, **kwargs):
        """
        Run a TWISS on the model.

        :param str sequence: sequence name
        :param str range: range name
        :param kwargs: further keyword arguments for the MAD-X command
        """
        _sequence = self.set_sequence(sequence, range)
        _range = self._get_range_bounds()
        kw = self.get_twiss_initial().copy()
        kw.update(kwargs)
        result = self.madx.twiss(sequence=_sequence, range=_range, **kw)
        if _range == ('#s', '#e'):
            # we say that when the "full" range has been selected,
            # we can set this to true. Needed for e.g. aperture calls
            self._twiss_called[sequence] = True
        return result

    def survey(self, sequence=None, range=None, **kwargs):
        """
        Run a survey on the model.

        :param str sequence: sequence name
        :param str range: range name
        :param kwargs: further keyword arguments for the MAD-X command
        """
        _sequence = self.set_sequence(sequence, range)
        _range = self._get_range_bounds()
        kw = self.get_twiss_initial().copy()
        kw.update(kwargs)
        return self.madx.survey(sequence=_sequence, range=_range, **kw)

    def aperture(self,
               sequence=None,
               range=None,
               columns='name,l,s,n1,aper_1,aper_2,aper_3,aper_4',
               **kwargs):
        """
        Get the aperture from the model.

        :param str sequence: sequence name
        :param str range: range name
        :param kwargs: further keyword arguments for the MAD-X command
        """
        _sequence = self.set_sequence(sequence, range)
        _range = self._get_range_bounds()
        if not self._twiss_called.get(sequence):
            self.twiss(sequence)
        # call "basic aperture files"
        if not self._aperture_called.get(_sequence):
            self._call(*self._active_sequence_def['aperfiles'])
            self._aperture_called[sequence] = True
        # getting offset file if any:
        # if no range was selected, we ignore offsets...
        if range and 'aper-offset' in self._active_range_def:
            offsets_file = self.mdata.get_by_dict(rangedict['aper-offset'])
            with offsets_file.filename() as _offsets:
                return self._madx.aperture(sequence=_sequence, range=_range,
                                           offsets=_offsets, **kwargs)
        else:
            return self._madx.aperture(sequence=_sequence, range=_range,
                                       **kwargs)

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

        _sequence = self.set_sequence(sequence)
        _range = self._get_range_bounds()

        seqdict = self._mdef['sequences'][sequence]
        rangedict = seqdict['ranges'][_range]

        def is_match_param(v):
            return v.lower() in ['rmatrix', 'chrom', 'beta0', 'deltap',
                    'betx','alfx','mux','x','px','dx','dpx',
                    'bety','alfy','muy','y','py','dy','dpy' ]

        if 'twiss-initial-conditions' in rangedict:
            twiss_init = dict(
                (key, val)
                for key, val in self._get_twiss_initial(sequence, _range).items()
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

    # INTERNALS:

    def _call(self, *files):
        for file in *files:
            with self._repo.get(file).filename() as fpath:
                self.call(fpath)

    def _get_sequence_def(self, sequence):
        """
        Return the sequence dictionary in the model definition.

        :param str sequence: sequence name
        :returns: sequence definition
        :rtype: dict
        :raises KeyError: if the sequence is not defined
        """
        return self._mdef['sequences'][sequence or self._active_sequence]

    def _get_range_def(self, sequence=None, range=None):
        """
        Return the range dictionary in the model definition.

        :param str sequence: sequence name
        :param str range: range name
        :returns: range definition
        :rtype: dict
        :raises KeyError: if sequence or range is not defined
        """
        seqdict = self._get_sequence_def(sequence)
        return seqdict['ranges'][range or self._active_range]

    def _get_twiss_initial(self, sequence=None, range=None, name=None):
        """
        Return the twiss initial conditions.

        :param str sequence:
        :param str range:
        :param str name:
        :raises KeyError:
        """
        rangedict = self._get_range_def(sequence=sequence, range=_range)
        _name = name or rangedict['default-twiss']
        return rangedict['twiss-initial-conditions'][name] # raises

    def _get_optic_def(self, optic):
        """
        Return optic data from model definition.

        :param str optic: optic name
        :returns: optic information in model definition
        :rtype: dict
        :raises KeyError: if the optic is not defined
        """
        return self._mdef['optics'][optic]

    def _get_range_bounds(self):
        """
        Return MAD-X range of the currently selected sequence/range.

        :returns: range as defined in the model
        :rtype: tuple
        """
        rangedict = self._get_range_def()
        return (rangedict["madx-range"]["first"],
                rangedict["madx-range"]["last"])

    def _get_range_dict(self, sequence=None, range=None):
        """
        Returns the range dictionary. If sequence/range isn't given,
        returns default for the model
        """
        if not sequence:
            sequence = self._active_sequence
        elif sequence not in self._mdef['sequences']:
            raise ValueError("%s is not a valid sequence name, available sequences: '%s'" % (sequence, "' '".join(self._mdef['sequences'].keys())))

        seqdict = self._mdef['sequences'][sequence]
        if range:
            self.set_range(range)
        return seqdict['ranges'][self._active_range]


class Factory(object):

    """Model instance factory."""

    def __init__(self, model_locator, model_cls=None):
        """Create Model factory using a specified ModelLocator."""
        self._model_locator = model_locator
        self._model_cls = model_cls or Model

    def get_model_names(self):
        """Get iterable over all model names."""
        return self._model_locator.list_models()

    def _create(self, name, mdef, repo, sequence, optics, madx, command_log, logger):
        """
        Create Model instance based on ModelData.

        Parameters as in load_model (except for mdata).
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        if madx is None:
            madx = Madx(command_log=command_log, error_log=logger)
            madx.verbose(False)
        elif command_log is not None:
            raise ValueError("'command_log' cannot be used with 'madx'")
        return self._model_cls(name, mdef, repo,
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
                   command_log=None,
                   logger=None):
        """
        Find model definition by name and create Model instance.

        :param str name: model name
        :param str sequence: Name of the initial sequence to use
        :param str optics: Name of optics to load, string or list of strings.
        :param Madx madx: MAD-X instance to use
        :param str command_log: history file name; use only if madx is None!
        :param logging.Logger logger:
        """
        mdef = self._model_locator.get_definition(name)
        repo = self._model_locator.get_repository(mdef)
        return self._create(name,
                            mdef,
                            repo,
                            sequence=sequence,
                            optics=optics,
                            madx=madx,
                            command_log=command_log,
                            logger=logger)


class Locator(ModelLocator):

    """
    Model locator for yaml files that contain multiple model definitions.

    These are the model definition files that are currently used by default
    for filesystem resources.

    Serves the purpose of locating models and returning corresponding
    resource providers.
    """

    def __init__(self, resource_provider):
        """
        Initialize a merged model locator instance.

        The resource_provider parameter must be a ResourceProvider instance
        that points to the filesystem location where the .cpymad.yml model
        files are stored.
        """
        self.res_provider = resource_provider

    def list_models(self, encoding='utf-8'):
        """
        Iterate all available models.

        Returns an iterable that may be a generator object.
        """
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.yml'):
            mdefs = self.res_provider.yaml(res_name, encoding=encoding)
            for n, d in mdefs.items():
                if d['real']:
                    yield n

    def get_definition(self, name, encoding='utf-8'):
        """
        Get the first found model with the specified name.

        :returns: the model definition
        :raises ValueError: if no model with the given name is found.
        """
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.yml'):
            mdefs = self.res_provider.yaml(res_name, encoding=encoding)
            # restrict only to 'real' models (don't do that?):
            if name in (n for n, d in mdefs.items() if d['real']):
                break
        else:
            raise ValueError("The model "+name+" does not exist in the database")

        # expand the model using its bases specified in 'extends'. try to
        # provide a useful MRO:
        def get_bases(model_name):
            return mdefs[model_name].get('extends', [])
        mro = util.C3_mro(get_bases, name)

        # TODO: this could be done using some sort of ChainMap, i.e.
        # merging at lookup time instead of at creation time. but this is
        # probably not worth the trouble for now.
        real_mdef = {}
        for base in reversed(mro):
            util.deep_update(real_mdef, mdefs[base])

        return real_mdef

    def get_repository(self, mdef):
        """
        Get the resource loader for the given model.
        """
        # instantiate the resource providers for model resource data
        repo_offs = mdef['path-offsets']['repository-offset']
        # the repository location may be overwritten by dbdirs:
        for dbdir in mdef.get('dbdirs', []):
            if os.path.isdir(dbdir):
                return FileResource(os.path.join(dbdir, repo_offs))
        return self.res_provider.get('repdata').get(repo_offs)
