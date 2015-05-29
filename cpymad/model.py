"""
Models encapsulate metadata for accelerator machines.

For more information about models, see :class:`Model`.
"""

from __future__ import absolute_import

import logging
import os

from .madx import Madx
from .util import is_match_param
from .resource.file import FileResource


__all__ = [
    'Model',
    'Beam',
    'Optic',
    'Sequence',
    'Range',
    'Locator',
]


def _deserialize(data, cls, *args):
    """Create an instance dictionary from a data dictionary."""
    return {key: cls(key, val, *args) for key, val in data.items()}


def _serialize(data):
    """Create a data dictionary from an instance dictionary."""
    return {key: val.data for key, val in data.items()}


class Model(object):

    """
    A model is a complete description of an accelerator machine.

    This class is used to bundle all metadata related to an accelerator and
    all its configurations. It takes care of loading the proper MAD-X files
    when needed. Models are conceptually derived from the JMad models, but
    have evolved to a more pythonic and simple API.

    To create a model instance from a model definition file, use the
    ``Model.load`` constructor.

    Only GET access is allowed to all instance variables at the moment.

    Public attributes:

    :ivar str name: model name
    :ivar dict beams: known :class:`Beam` objects
    :ivar dict optics: known :class:`Optic` objects
    :ivar dict sequences: known :class:`Sequence` objects
    :ivar Madx madx: handle to the MAD-X library

    Private variables:

    :ivar dict _data: model definition data
    :ivar ResourceProvider _repo: resource access

    The following example demonstrates the basic usage:

    .. code-block:: python

        from cpymad.model import Model

        model = Model.load('/path/to/model/definition.cpymad.yml')

        twiss = model.default_sequence.twiss()

        print("max/min beta x:", max(twiss['betx']), min(twiss['betx']))
        print("ex: {0}, ey: {1}", twiss.summary['ex'], twiss.summary['ey'])
    """

    # current version of model API
    API_VERSION = 0

    def __init__(self, data, repo, madx):
        """
        Initialize a Model object.

        :param dict data: model definition data
        :param ResourceProvider repo: resource repository
        :param Madx madx: MAD-X instance to use
        """
        self.check_compatibility(data)
        # init instance variables
        self._data = data
        self._repo = repo
        self.madx = madx
        self._loaded = False
        # create Beam/Optic/Sequence instances:
        self.beams = _deserialize(data['beams'], Beam, self)
        self.optics = _deserialize(data['optics'], Optic, self)
        self.sequences = _deserialize(data['sequences'], Sequence, self)

    @classmethod
    def check_compatibility(cls, data):
        """
        Check a model definition for compatibility.

        :param dict data: a model definition to be tested
        :raises ValueError: if the model definition is incompatible
        """
        model_api = data.get('api_version', 'undefined')
        if model_api != cls.API_VERSION:
            raise ValueError(("Incompatible model API version: {!r},\n"
                              "              Required version: {!r}")
                             .format(model_api, cls.API_VERSION))

    @classmethod
    def load(cls,
             name,
             # *,
             # These should be passed as keyword-only parameters:
             locator=None,
             madx=None,
             command_log=None,
             error_log=None):
        """
        Create Model instance from a model definition file.

        :param str name: model definition file name
        :param Locator locator: model locator
        :param Madx madx: MAD-X instance to use
        :param str command_log: history file name; use only if madx is None!
        :param logging.Logger error_log:

        If the ``locator`` is not specified ``name`` is assumed to be an
        absolute path of a model definition file living in the ordinary file
        system.
        """
        if locator is None:
            path, name = os.path.split(name)
            locator = Locator(FileResource(path))
        data = locator.get_definition(name)
        repo = locator.get_repository(data)
        if madx is None:
            if error_log is None:
                error_log = logging.getLogger(__name__ + '.' + name)
            madx = Madx(command_log=command_log, error_log=error_log)
            madx.verbose(False)
        elif command_log is not None:
            raise ValueError("'command_log' cannot be used with 'madx'")
        elif error_log is not None:
            raise ValueError("'error_log' cannot be used with 'madx'")
        model = cls(data, repo=repo, madx=madx)
        model.init()
        return model

    def init(self):
        """Load model in MAD-X interpreter."""
        if self._loaded:
            return
        self._loaded = True
        self._load(*self._data['init-files'])

    def __repr__(self):
        return "{0}({1!r})".format(self.__class__.__name__, self.name)

    @property
    def name(self):
        """Model name."""
        return self._data['name']

    @property
    def data(self):
        """Get a serializable representation of this model."""
        data = self._data.copy()
        data['beams'] = _serialize(self.beams)
        data['optics'] = _serialize(self.optics)
        data['sequences'] = _serialize(self.sequences)
        return data

    @property
    def default_optic(self):
        """Get default Optic."""
        return self.optics[self._data['default-optic']]

    @property
    def default_sequence(self):
        """Get default Sequence."""
        return self.sequences[self._data['default-sequence']]

    # TODO: add setters for default_optic / default_sequence
    # TODO: remove default_sequence?

    def _load(self, *files):
        """Load MAD-X files in interpreter."""
        for file in files:
            with self._repo.get(file).filename() as fpath:
                self.madx.call(fpath)


class Beam(object):

    """
    Beam for :class:`Model`.

    A beam defines the mass, charge, energy, etc. of the particles moved
    through the accelerator.

    Public attributes:

    :ivar str name: beam name
    :ivar dict data: beam parameters (keywords to BEAM command in MAD-X)

    Private variables:

    :ivar Model _model: owning model
    :ivar bool _loaded: beam has been initialized in MAD-X
    """

    def __init__(self, name, data, model):
        """Initialize instance variables."""
        self.name = name
        self.data = data
        self._model = model
        self._loaded = False

    def init(self):
        """Define the beam in MAD-X."""
        if self._loaded:
            return
        self._loaded = True
        self._model.init()
        self._model.madx.command.beam(**self.data)


class Optic(object):

    """
    Optic for :class:`Model`.

    An optic (as far as I understand) defines a variant of the accelerator
    setup, e.g. different injection mechanisms.

    Public attributes:

    :ivar str name: optic name

    Private variables:

    :ivar dict _data: optic definition
    :ivar Model _model: owning model
    :ivar bool _loaded: beam has been initialized in MAD-X
    """

    def __init__(self, name, data, model):
        """Initialize instance variables."""
        self.name = name
        self.data = data
        self._model = model
        self._loaded = False

    def init(self):
        """Load the optic in the MAD-X process."""
        if self._loaded:
            return
        self._loaded = True
        self._model.init()
        self._model._load(*self.data.get('init-files', ()))


class Sequence(object):

    """
    Sequence for :class:`Model`.

    A sequence defines an arrangement of beam line elements. It can be
    subdivided into multiple ranges.

    Public attributes:

    :ivar str name: sequence name
    :ivar dict ranges: known :class:`Range` objects

    Private variables:

    :ivar dict _data:
    :ivar Model _model:
    """

    def __init__(self, name, data, model):
        """Initialize instance variables."""
        self.name = name
        self._data = data
        self._model = model
        self.ranges = _deserialize(data['ranges'], Range, self)

    def init(self):
        """Load model in MAD-X interpreter."""
        self._model.init()
        self.beam.init()

    @property
    def data(self):
        """Get a serializable representation of this sequence."""
        data = self._data.copy()
        data['ranges'] = _serialize(self.ranges)
        return data

    @property
    def beam(self):
        """Get :class:`Beam` instance for this sequence."""
        return self._model.beams[self._data['beam']]

    @property
    def default_range(self):
        """Get default :class:`Range`."""
        return self.ranges[self._data['default-range']]

    @property
    def real_sequence(self):
        """Get the corresponding :class:`cpymad.madx.Sequence`."""
        self.init()
        return self._model.madx.sequences[self.name]

    @property
    def elements(self):
        """Get a proxy list for all the elements."""
        return self.real_sequence.elements

    def range(self, start, stop):
        """Create a :class:`Range` within (start, stop) for this sequence."""
        # TODO
        raise NotImplementedError()

    # MAD-X commands:

    def twiss(self, **kwargs):
        """Execute a TWISS command on the default range."""
        return self.default_range.twiss(**kwargs)

    def survey(self, **kwargs):
        """Run SURVEY on this sequence."""
        self.init()
        return self._model.madx.survey(sequence=self.name, **kwargs)

    def match(self, **kwargs):
        """Run MATCH on this sequence."""
        return self.default_range.match(**kwargs)


class Range(object):

    """
    Range for :class:`Model`.

    A range is a subsequence of elements within a :class:`Sequence`.

    Public attributes:

    :ivar str name: sequence name

    Private variables:

    :ivar dict _data:
    :ivar Sequence _sequence:
    """

    def __init__(self, name, data, sequence):
        """Initialize instance variables."""
        self.name = name
        self.data = data
        self._sequence = sequence

    def init(self):
        """Load model in MAD-X interpreter."""
        self._sequence.init()

    @property
    def bounds(self):
        """Get a tuple (first, last)."""
        return (self.data["madx-range"]["first"],
                self.data["madx-range"]["last"])

    @property
    def offsets_file(self):
        """Get a :class:`ResourceProvider` for the offsets file."""
        if 'aper-offset' not in self.data:
            return None
        repo = self._sequence._model._repo
        return _repo.get(self.data['aper-offset'])

    def twiss(self, **kwargs):
        """Run TWISS on this range."""
        self.init()
        kw = self._set_twiss_init(kwargs)
        madx = self._sequence._model.madx
        result = madx.twiss(sequence=self._sequence.name,
                            range=self.bounds, **kw)
        return result

    def match(self, **kwargs):
        """Perform a MATCH operation on this range."""
        self.init()
        kw = self._set_twiss_init(kwargs)
        kw['twiss_init'] = {
            key: val
            for key, val in kw['twiss_init'].items()
            if is_match_param(key)
        }
        madx = self._sequence._model.madx
        return madx.match(sequence=self._sequence.name,
                          range=self.bounds, **kw)

    @property
    def initial_conditions(self):
        """
        Return a dict of all defined initial conditions.

        Each item is a dict of TWISS parameters.
        """
        return self.data['twiss-initial-conditions']

    @property
    def default_initial_conditions(self):
        """Return the default twiss initial conditions."""
        return self.initial_conditions[self.data['default-twiss']]

    def _set_twiss_init(self, kwargs):
        kw = kwargs.copy()
        twiss_init = kw.get('twiss_init', {}).copy()
        twiss_init.update(self.default_initial_conditions)
        kw['twiss_init'] = twiss_init
        return kw


class Locator(object):

    """
    Model locator for yaml files that contain multiple model definitions.

    These are the model definition files that are currently used by default
    for filesystem resources.

    Serves the purpose of locating models and returning corresponding
    resource providers.
    """

    ext = '.cpymad.yml'

    def __init__(self, resource_provider):
        """
        Initialize a merged model locator instance.

        The resource_provider parameter must be a ResourceProvider instance
        that points to the filesystem location where the .cpymad.yml model
        files are stored.
        """
        self._repo = resource_provider

    def list_models(self, encoding='utf-8'):
        """
        Iterate all available models.

        Returns an iterable that may be a generator object.
        """
        for res_name in self._repo.listdir_filter(ext=self.ext):
            yield res_name[:-len(self.ext)]

    def get_definition(self, name, encoding='utf-8'):
        """
        Get the first found model with the specified name.

        :returns: the model definition
        :raises ValueError: if no model with the given name is found.
        """
        try:
            if not name.endswith(self.ext):
                name += self.ext
            return self._repo.yaml(name, encoding=encoding)
        except IOError:
            raise ValueError("The model {!r} does not exist in the database"
                             .format(name))

    def get_repository(self, data):
        """
        Get the resource loader for the given model.
        """
        # instantiate the resource providers for model resource data
        return self._repo.get(data['path-offset'])
