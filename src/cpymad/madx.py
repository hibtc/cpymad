"""
This module defines a convenience layer to access the MAD-X interpreter.

The most interesting class for users is :class:`Madx`.
"""

from __future__ import absolute_import

from contextlib import contextmanager, suppress
from functools import wraps
from itertools import product
from numbers import Number
import collections.abc as abc
import os
import subprocess
import sys

import numpy as np

from . import _rpc
from . import util
from .stream import AsyncReader, TextCallback


__all__ = [
    'Madx',
    'ArrayAttribute',
    'AttrDict',
    'BaseTypeMap',
    'Command',
    'CommandLog',
    'CommandMap',
    'Element',
    'ElementList',
    'ExpandedElementList',
    'GlobalElementList',
    'Metadata',
    'Sequence',
    'SequenceMap',
    'Table',
    'TableMap',
    'VarList',
    'VarParamList',
    'Version',

    # Data
    'metadata',

    # Errors:
    'TwissFailed',
]


class TwissFailed(RuntimeError):
    pass


class Version:

    """Version information struct. """

    def __init__(self, release, date):
        """Store version information."""
        self.release = release
        self.date = date
        self.info = tuple(map(int, release.split('.')))

    def __repr__(self):
        """Show nice version string to user."""
        return "MAD-X {} ({})".format(self.release, self.date)


class NullContext:
    __enter__ = __exit__ = lambda *_: None


class CommandLog:

    """Log MAD-X command history to a text file."""

    @classmethod
    def create(cls, filename, prefix='', suffix='\n'):
        """Create CommandLog from filename (overwrite/create)."""
        return cls(open(filename, 'wt'), prefix, suffix, own=True)

    def __init__(self, file, prefix='', suffix='\n', own=False):
        """Create CommandLog from file instance."""
        self._file = file
        self._prefix = prefix
        self._suffix = suffix
        self._own = own

    def __del__(self):
        self.close()

    def __call__(self, command: str):
        """Log a single history line and flush to file immediately."""
        self._file.write(self._prefix + command + self._suffix)
        self._file.flush()

    def close(self):
        if self._own:
            self._file.close()


class Madx:

    """
    Python interface for a MAD-X process.

    For usage instructions, please refer to:

        https://hibtc.github.io/cpymad/getting-started

    Communicates with a MAD-X interpreter in a background process.

    The state of the MAD-X interpreter is controlled by feeding textual MAD-X
    commands to the interpreter.

    The state of the MAD-X interpreter is accessed by directly reading the
    values from the C variables in-memory and sending the results pickled back
    over the pipe.

    Data attributes:

    :ivar command:      Mapping of all MAD-X commands.
    :ivar globals:      Mapping of global MAD-X variables.
    :ivar elements:     Mapping of globally visible elements.
    :ivar base_types:   Mapping of MAD-X base elements.
    :ivar sequence:     Mapping of all sequences in memory.
    :ivar table:        Mapping of all tables in memory.
    """

    def __init__(self, libmadx=None, command_log=None, stdout=None,
                 history=None, prompt=None, **Popen_args):
        """
        Initialize instance variables.

        :param libmadx: :mod:`libmadx` compatible object
        :param command_log: Log all MAD-X commands issued via cpymad.
        :param stdout: file descriptor, file object or callable
        :param str prompt: prefix for a new :class:`CommandLog`
        :param Popen_args: Additional parameters to ``subprocess.Popen``

        If ``libmadx`` is NOT specified, a new MAD-X interpreter will
        automatically be spawned. This is what you will mostly want to do. In
        this case any additional keyword arguments are forwarded to
        ``subprocess.Popen``. The most prominent use case for this is to
        redirect or suppress the MAD-X standard I/O::

            m = Madx(stdout=False)

            with open('madx_output.log', 'w') as f:
                m = Madx(stdout=f)

            m = Madx(stdout=sys.stdout)
        """
        if isinstance(command_log, str):
            # open new history file:
            command_log = CommandLog.create(command_log, prompt or '')
        elif hasattr(command_log, 'write'):
            # assuming stream already opened:
            command_log = CommandLog(command_log, prompt or '')
        elif prompt is not None:
            assert command_log is None, \
                "Passing fully constructed `command_log` instances is " \
                "incompatible with parameter `prompt`."
            command_log = CommandLog(sys.stdout, prompt)
        self.reader = NullContext()
        # start libmadx subprocess
        if libmadx is None:
            if stdout is None:
                stdout = sys.stdout
            if hasattr(stdout, 'write'):
                # Detect if stdout is attached to a jupyter notebook:
                cls = getattr(stdout, '__class__', type(None))
                qualname = cls.__module__ + '.' + cls.__name__
                if qualname == 'ipykernel.iostream.OutStream':
                    # In that case we want to behave the same way as `print`
                    # (i.e. log to the notebook not to the terminal). On
                    # linux, python>=3.7 within notebooks 6.4 `sys.stdout` has
                    # a valid sys.stdout.fileno(), but writing to it outputs
                    # to the terminal, so we have to use `sys.stdout.write()`:
                    stdout = stdout.write
                else:
                    # Otherwise, let the OS handle MAD-X output, by passing
                    # the file descriptor if available. This is preferred
                    # because it is faster, and also because it means that the
                    # MAD-X output is directly connected to the output as
                    # binary stream, without potential recoding errors.
                    try:
                        stdout = stdout.fileno()
                    except (AttributeError, OSError, IOError):
                        stdout = stdout.write
                # Check for text stream to prevent TypeError (see #110).
                if callable(stdout):
                    try:
                        stdout(b'')
                    except (TypeError, ValueError):
                        stdout = TextCallback(stdout)
            Popen_args['stdout'] = \
                subprocess.PIPE if callable(stdout) else stdout
            # stdin=None leads to an error on windows when STDIN is broken.
            # Therefore, we need set stdin=os.devnull by passing stdin=False:
            Popen_args.setdefault('stdin', False)
            Popen_args.setdefault('bufsize', 0)
            self._service, self._process = \
                _rpc.LibMadxClient.spawn_subprocess(**Popen_args)
            libmadx = self._service.libmadx
            if callable(stdout):
                self.reader = AsyncReader(self._process.stdout, stdout)
        if not libmadx.is_started():
            with self.reader:
                libmadx.start()
        # init instance variables:
        self.history = history
        self._libmadx = libmadx
        self._command_log = command_log
        self.command = CommandMap(self)
        self.globals = VarList(self)
        self.elements = GlobalElementList(self)
        self.base_types = BaseTypeMap(self)
        self.sequence = SequenceMap(self)
        self.beams = BeamMap(self)
        self.table = TableMap(self._libmadx)
        self._enter_count = 0
        self._batch = None

    def __bool__(self):
        """Check if MAD-X is up and running."""
        try:
            libmadx = self._libmadx
            # short-circuit implemented in minrpc.client.RemoteModule:
            return bool(libmadx) and libmadx.is_started()
        except (_rpc.RemoteProcessClosed, _rpc.RemoteProcessCrashed):
            return False

    def __getattr__(self, name):
        """Resolve missing attributes as commands."""
        try:
            return getattr(self.command, name)
        except AttributeError:
            raise AttributeError(
                'Unknown attribute or command: {!r}'.format(name)) from None

    def quit(self):
        """Shutdown MAD-X interpreter and stop process."""
        with suppress(AttributeError, RuntimeError):
            self.input('quit;')
        with suppress(AttributeError, RuntimeError):
            self._service.close()
        with suppress(AttributeError, RuntimeError):
            self._process.wait()
        if hasattr(self._command_log, 'close'):
            self._command_log.close()
            self._command_log = None

    exit = quit

    def __enter__(self):
        """Use as context manager to ensure that MAD-X is terminated."""
        return self

    def __exit__(self, *exc_info):
        self.quit()

    # Data descriptors:

    @property
    def version(self) -> Version:
        """Get the MAD-X version."""
        return Version(self._libmadx.get_version_number(),
                       self._libmadx.get_version_date())

    @property
    def options(self) -> "Command":
        """Values of current options."""
        return Command(self, self._libmadx.get_options())

    @property
    def beam(self):
        """Get the current default beam."""
        return Command(self._madx, self._libmadx.get_current_beam())

    # Methods:

    def input(self, text: str) -> bool:
        """
        Run any textual MAD-X input.

        :param text: command text
        :returns: whether the command has completed without error
        """
        text = text.rstrip(';') + ';'
        if self._enter_count > 0:
            self._batch.append(text)
            return True
        # write to history before performing the input, so if MAD-X
        # crashes, it is easier to see, where it happened:
        if self.history is not None:
            self.history.append(text)
        if self._command_log:
            self._command_log(text)
        try:
            with self.reader:
                return self._libmadx.input(text)
        except _rpc.RemoteProcessCrashed:
            raise RuntimeError("MAD-X has stopped working!") from None

    __call__ = input

    @contextmanager
    def batch(self):
        """
        Collect input and send in a single batch when leaving context. This is
        useful to improve performance when issueing many related commands in
        quick succession.

        Example:

        >>> with madx.batch():
        ...     madx.globals.update(optic)
        """
        self._enter_count += 1
        if self._enter_count == 1:
            self._batch = []
        try:
            yield None
        finally:
            self._enter_count -= 1
            if self._enter_count == 0:
                self.input("\n".join(self._batch))
                self._batch = None

    def expr_vars(self, expr: str) -> list:
        """Find all variable names used in an expression. This does *not*
        include element attribute nor function names."""
        if not isinstance(expr, str):
            return []
        return [v for v in util.expr_symbols(expr)
                if util.is_identifier(v)
                and v in self.globals
                and self._libmadx.get_var_type(v) > 0]

    def chdir(self, dir: str) -> util.ChangeDirectory:
        """
        Change the directory of the MAD-X process (not the current python process).

        :param dir: new path name
        :returns: a context manager that can change the directory back

        It can be used as context manager for temporary directory changes::

            with madx.chdir('/x/y/z'):
                madx.call('file.x')
                madx.call('file.y')
        """
        return util.ChangeDirectory(dir, self._chdir, self._libmadx.getcwd)

    # Turns `dir` into a keyword argument for CHDIR command:
    def _chdir(self, dir: str):
        return self.command.chdir(dir=dir)

    def call(self, file: str, chdir: bool = False):
        """
        CALL a file in the MAD-X interpreter.

        :param file: file name with path
        :param chdir: temporarily change directory in MAD-X process
        """
        if chdir:
            dirname, basename = os.path.split(file)
            with self.chdir(dirname):
                self.command.call(file=basename)
        else:
            self.command.call(file=file)

    def twiss(self, **kwargs):
        """
        Run TWISS.

        :param str sequence: name of sequence
        :param kwargs: keyword arguments for the MAD-X command

        Note that the kwargs overwrite any arguments in twiss_init.
        """
        if not self.command.twiss(**kwargs):
            raise TwissFailed()
        table = kwargs.get('table', 'twiss')
        if 'file' not in kwargs:
            self._libmadx.apply_table_selections(table)
        return self.table[table]

    def survey(self, **kwargs):
        """
        Run SURVEY.

        :param str sequence: name of sequence
        :param kwargs: keyword arguments for the MAD-X command
        """
        self.command.survey(**kwargs)
        table = kwargs.get('table', 'survey')
        if 'file' not in kwargs:
            self._libmadx.apply_table_selections(table)
        return self.table[table]

    def use(self, sequence: str = None, range: str = None, **kwargs):
        """
        Run USE to expand a sequence.

        :param str sequence: sequence name
        :returns: name of active sequence
        """
        self.command.use(sequence=sequence, range=range, **kwargs)

    def sectormap(self, elems, **kwargs):
        """
        Compute the 7D transfer maps (the 7'th column accounting for KICKs)
        for the given elements and return as Nx7x7 array.
        """
        self.command.select(flag='sectormap', clear=True)
        for elem in elems:
            self.command.select(flag='sectormap', range=elem)
        with util.temp_filename() as sectorfile:
            self.twiss(sectormap=True, sectorfile=sectorfile, **kwargs)
        return self.sectortable(kwargs.get('sectortable', 'sectortable'))

    def sectortable(self, name='sectortable'):
        """Read sectormap + kicks from memory and return as Nx7x7 array."""
        tab = self.table[name]
        cnt = len(tab['r11'])
        return np.vstack((
            np.hstack((tab.rmat(slice(None)),
                       tab.kvec(slice(None)).reshape((6, 1, -1)))),
            np.hstack((np.zeros((1, 6, cnt)),
                       np.ones((1, 1, cnt)))),
        )).transpose((2, 0, 1))

    def sectortable2(self, name='sectortable'):
        """Read 2nd order sectormap T_ijk, return as Nx6x6x6 array."""
        tab = self.table[name]
        return tab.tmat(slice(None)).transpose((3, 0, 1, 2))

    def match(self,
              constraints=[],
              vary=[],
              weight=None,
              method=('lmdif', {}),
              knobfile=None,
              limits=None,
              **kwargs) -> dict:
        """
        Perform a simple MATCH operation.

        For more advanced cases, you should issue the commands manually.

        :param list constraints: constraints to pose during matching
        :param list vary: knob names to be varied
        :param dict weight: weights for matching parameters
        :param str knobfile: file to write the knob values to
        :param dict kwargs: keyword arguments for the MAD-X command
        :returns: final knob values

        Example:

        >>> from cpymad.madx import Madx
        >>> from cpymad.types import Constraint
        >>> m = Madx()
        >>> m.call('sequence.madx')
        >>> twiss_init = {'betx': 1, 'bety': 2, 'alfx': 3, 'alfy': 4}
        >>> m.match(
        ...     sequence='mysequence',
        ...     constraints=[
        ...         dict(range='marker1',
        ...              betx=Constraint(min=1, max=3),
        ...              bety=2)
        ...     ],
        ...     vary=['qp1->k1',
        ...           'qp2->k1'],
        ...     **twiss_init,
        ... )
        >>> tw = m.twiss('mysequence', **twiss_init)
        """
        command = self.command
        # MATCH (=start)
        command.match(**kwargs)
        if weight:
            command.weight(**weight)
        for c in constraints:
            command.constraint(**c)
        limits = limits or {}
        for v in vary:
            command.vary(name=v, **limits.get(v, {}))
        command[method[0]](**method[1])
        command.endmatch(knobfile=knobfile)
        return dict((knob, self.eval(knob)) for knob in vary)

    def verbose(self, switch=True):
        """Turn verbose output on/off."""
        self.command.option(echo=switch, warn=switch, info=switch)

    def eval(self, expr) -> float:
        """
        Evaluates an expression and returns the result as double.

        :param str expr: expression to evaluate.
        :returns: numeric value of the expression
        """
        if isinstance(expr, (float, int, bool)):
            return expr
        if isinstance(expr, (list, ArrayAttribute)):
            return [self.eval(x) for x in expr]
        return self._libmadx.eval(expr)


class _Mapping(abc.Mapping):

    def __repr__(self):
        """String representation of a custom mapping object."""
        return str(dict(self))

    def __str__(self):
        return repr(self)

    def __getattr__(self, key):
        key = util._fix_name(key)
        try:
            return self[key]
        except KeyError:
            raise AttributeError(self._missing(key)) from None

    def _missing(self, key):
        return key


class _MutableMapping(_Mapping, abc.MutableMapping):

    __slots__ = ()

    def __setattr__(self, key, val):
        if key in self.__slots__:
            object.__setattr__(self, key, val)
        else:
            key = util._fix_name(key)
            self[key] = val

    def __delattr__(self, key):
        if key in self.__slots__:
            object.__delattr__(self, key)
        else:
            key = util._fix_name(key)
            del self[key]


class AttrDict(_Mapping):

    def __init__(self, data):
        if not isinstance(data, abc.Mapping):
            data = dict(data)
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, name):
        return self._data[name.lower()]

    def __contains__(self, name):
        return name.lower() in self._data

    def __len__(self):
        return len(self._data)

    def update(self, *args, **kwargs):
        self._data.update(*args, **kwargs)


class SequenceMap(_Mapping):

    """Mapping of all sequences (:class:`Sequence`) in memory."""

    def __init__(self, madx):
        self._madx = madx
        self._libmadx = madx._libmadx

    def __iter__(self):
        return iter(self._libmadx.get_sequence_names())

    def __getitem__(self, name):
        try:
            return Sequence(name, self._madx)
        except ValueError:
            raise KeyError("Unknown sequence: {!r}".format(name)) from None

    def __contains__(self, name):
        return self._libmadx.sequence_exists(name.lower())

    def __len__(self):
        return self._libmadx.get_sequence_count()

    def __call__(self):
        """The active :class:`Sequence` (may be None)."""
        try:
            return Sequence(self._libmadx.get_active_sequence_name(),
                            self._madx, _check=False)
        except RuntimeError:
            return None


class BeamMap(_Mapping):
    """Mapping of all beams (:class:`Beam`) in memory."""

    def __init__(self, madx):
        self._madx = madx
        self._libmadx = madx._libmadx

    def __iter__(self):
        return iter(self._libmadx.get_beam_names())

    def __getitem__(self, name):
        try:
            return Command(self._madx, self._libmadx.get_beam(name))
        except ValueError:
            raise KeyError("Unknown beam: {!r}".format(name)) from None

    def __contains__(self, name):
        return name in self._libmadx.get_beam_names()

    def __len__(self):
        return len(self._libmadx.get_beam_names())


class TableMap(_Mapping):

    """Mapping of all tables (:class:`Table`) in memory."""

    def __init__(self, libmadx):
        self._libmadx = libmadx

    def __iter__(self):
        return iter(self._libmadx.get_table_names())

    def __getitem__(self, name):
        try:
            return Table(name, self._libmadx)
        except ValueError:
            raise KeyError("Table not found {!r}".format(name)) from None

    def __contains__(self, name):
        return self._libmadx.table_exists(name.lower())

    def __len__(self):
        return self._libmadx.get_table_count()


class Sequence:

    """
    MAD-X sequence representation.
    """

    def __init__(self, name, madx, _check=True):
        """Store sequence name."""
        self._name = name = name.lower()
        self._madx = madx
        self._libmadx = madx._libmadx
        if _check and not self._libmadx.sequence_exists(name):
            raise ValueError("Invalid sequence: {!r}".format(name))

    def __str__(self):
        """String representation."""
        return "<{}: {}>".format(self.__class__.__name__, self._name)

    def __eq__(self, other):
        """Comparison by sequence name."""
        if isinstance(other, Sequence):
            other = other.name
        return self.name == other

    __repr__ = __str__

    @property
    def name(self):
        """Get the name of the sequence."""
        return self._name

    @property
    def beam(self):
        """Get the beam dictionary associated to the sequence."""
        return Command(self._madx, self._libmadx.get_sequence_beam(self._name))

    @beam.setter
    def beam(self, beam):
        self._madx.command.beam(sequence=self._name, **beam)

    @property
    def twiss_table(self):
        """Get the TWISS results from the last calculation."""
        return Table(self.twiss_table_name, self._libmadx)

    @property
    def twiss_table_name(self):
        """Get the name of the table with the TWISS results."""
        return self._libmadx.get_sequence_twiss_table_name(self._name)

    @property
    def elements(self):
        """Get list of elements."""
        return ElementList(self._madx, self._name)

    def _get_length_parameter(self):
        """Return sequence length in the declaration"""
        return self._libmadx.get_sequence_length(self._name)

    @property
    def length(self):
        """Return sequence length in the declaration"""
        return self._get_length_parameter().value

    @property
    def expanded_elements(self):
        """List of elements including implicit drifts."""
        return ExpandedElementList(self._madx, self._name)

    def element_names(self):
        return self._libmadx.get_element_names(self._name)

    def element_positions(self):
        return self._libmadx.get_element_positions(self._name)

    def expanded_element_names(self):
        return self._libmadx.get_expanded_element_names(self._name)

    def expanded_element_positions(self):
        return self._libmadx.get_expanded_element_positions(self._name)

    @property
    def is_expanded(self):
        """Check if sequence is already expanded."""
        return self._libmadx.is_sequence_expanded(self._name)

    @property
    def has_beam(self):
        """Check if the sequence has an associated beam."""
        try:
            self.beam
            return True
        except RuntimeError:
            return False

    def expand(self):
        """Expand sequence (needed for expanded_elements)."""
        if self.is_expanded:
            return
        if not self.has_beam:
            self.beam = {}
        self.use()

    def use(self):
        """Set this sequence as active."""
        self._madx.use(self._name)


class Command(_MutableMapping):

    """
    Raw python interface to issue and view MAD-X commands. Usage example:

    >>> madx.command.twiss(sequence='LEBT')
    >>> madx.command.title('A meaningful phrase')
    >>> madx.command.twiss.betx
    0.0
    """

    __slots__ = ('_madx', '_data', '_attr', 'cmdpar')

    def __init__(self, madx, data):
        self._madx = madx
        self._data = data.pop('data')       # command parameters
        self._attr = data                   # further attributes
        self.cmdpar = AttrDict(self._data)

    def __repr__(self):
        """String representation as MAD-X statement."""
        overrides = {k: v.value for k, v in self._data.items() if v.inform}
        if self._attr.get('parent', self.name) == self.name:
            return util.format_command(self, **overrides)
        return self.name + ': ' + util.format_command(self.parent, **overrides)

    def __iter__(self):
        return iter(self._data)

    def __getattr__(self, name):
        try:
            return self._attr[name]
        except KeyError:
            return _Mapping.__getattr__(self, name)

    def __getitem__(self, name):
        try:
            return self._data[name.lower()].value
        except KeyError:
            raise KeyError("Unknown command: {!r}".format(name)) from None

    def __delitem__(self, name):
        raise NotImplementedError()

    def __setitem__(self, name, value):
        self(**{name: value})

    def __contains__(self, name):
        return name.lower() in self._data

    def __len__(self):
        return len(self._data)

    def __call__(*args, **kwargs):
        """Perform a single MAD-X command."""
        self, args = args[0], args[1:]
        if self.name == 'beam' and self.sequence:
            kwargs.setdefault('sequence', self.sequence)
        return self._madx.input(util.format_command(self, *args, **kwargs))

    def clone(*args, **kwargs):
        """
        Clone this command, assign the given name. This corresponds to the
        colon syntax in MAD-X, e.g.::

            madx.command.quadrupole.clone('qp', at=2, l=1)

        translates to the MAD-X command::

            qp: quadrupole, at=2, l=1;
        """
        self, name, args = args[0], args[1], args[2:]
        self._madx.input(
            name + ': ' + util.format_command(self, *args, **kwargs))
        return self._madx.elements.get(name)

    def _missing(self, key):
        return ('Unknown attribute {!r} for {!r} command!'
                .format(key, self.name))

    @property
    def defs(self):
        return AttrDict({
            key: par.definition
            for key, par in self.cmdpar.items()
        })


class Element(Command):

    def __getitem__(self, name):
        value = Command.__getitem__(self, name)
        if isinstance(value, list):
            return ArrayAttribute(self, value, name)
        return value

    def __delitem__(self, name):
        if self.parent is self:
            raise NotImplementedError(
                "Can't delete attribute {!r} in base element {!r}"
                .format(self.name, name))
        self[name] = self.parent[name]

    @property
    def parent(self):
        name = self._attr['parent']
        return (self if self.name == name else self._madx.elements[name])

    @property
    def base_type(self):
        name = self._attr['base_type']
        return (self if self.name == name else self._madx.elements[name])


class ArrayAttribute(abc.Sequence):

    def __init__(self, element, values, name):
        self._element = element
        self._values = values
        self._name = name

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value):
        if index >= len(self._values):
            self._values.extend([0] * (index - len(self._values) + 1))
        self._values[index] = value
        self._element[self._name] = self._values

    def __eq__(self, other):
        return self._values == other

    def __lt__(self, other):
        return self._values < other

    def __le__(self, other):
        return self._values <= other

    def __gt__(self, other):
        return self._values > other

    def __ge__(self, other):
        return self._values >= other

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return str(self._values)

    def __str__(self):
        return str(self._values)


class BaseElementList:

    """
    Immutable list of beam line elements.

    Each element is a dictionary containing its properties.
    """

    def __contains__(self, element):
        """Check if sequence contains element with specified name."""
        try:
            self.index(element)
            return True
        except ValueError:
            return False

    def __getitem__(self, index):
        """Return element with specified index."""
        if isinstance(index, str):
            # allow element names to be passed for convenience:
            try:
                index = self.index(index)
            except ValueError:
                raise KeyError(
                    "Unknown element: {!r}".format(index)) from None
        # _get_element accepts indices in the range [0, len-1]. The following
        # extends the accepted range to [-len, len+1], just like for lists:
        _len = len(self)
        if index < -_len or index >= _len:
            raise IndexError("Index out of bounds: {}, length is: {}"
                             .format(index, _len))
        if index < 0:
            index += _len
        data = self._get_element(index)
        data['index'] = index
        if data['base_type'] == 'sequence':
            return Sequence(data['name'], self._madx)
        else:
            return Element(self._madx, data)

    def __len__(self):
        """Get number of elements."""
        return self._get_element_count()

    def index(self, name):
        """
        Find index of element with specified name.

        :raises ValueError: if the element is not found
        """
        if len(self) == 0:
            raise ValueError('Empty element list.')
        name = name.lower()
        if name == '#s':
            return 0
        elif name == '#e':
            return len(self) - 1
        index = self._get_element_index(name)
        if index == -1:
            raise ValueError("Element not in list: {!r}".format(name))
        return index


class ElementList(BaseElementList, abc.Sequence):

    def __init__(self, madx, sequence_name):
        """
        Initialize instance.
        """
        self._madx = madx
        self._libmadx = madx._libmadx
        self._sequence_name = sequence_name

    def at(self, pos):
        """Find the element at specified S position."""
        return self._get_element_at(pos)

    def _get_element(self, element_index):
        return self._libmadx.get_element(self._sequence_name, element_index)

    def _get_element_count(self):
        return self._libmadx.get_element_count(self._sequence_name)

    def _get_element_index(self, element_name):
        return self._libmadx.get_element_index(self._sequence_name, element_name)

    def _get_element_at(self, pos):
        return self._libmadx.get_element_index_by_position(self._sequence_name, pos)

    def __repr__(self):
        return '[{}]'.format(', '.join(
            self._libmadx.get_element_names(self._sequence_name)))


class ExpandedElementList(ElementList):

    def _get_element(self, element_index):
        return self._libmadx.get_expanded_element(
            self._sequence_name, element_index)

    def _get_element_count(self):
        return self._libmadx.get_expanded_element_count(self._sequence_name)

    def _get_element_index(self, element_name):
        return self._libmadx.get_expanded_element_index(
            self._sequence_name, element_name)

    def _get_element_at(self, pos):
        return self._libmadx.get_expanded_element_index_by_position(
            self._sequence_name, pos)

    def __repr__(self):
        return '[{}]'.format(', '.join(
            self._libmadx.get_expanded_element_names(self._sequence_name)))


class GlobalElementList(BaseElementList, _Mapping):

    """Mapping of the global elements in MAD-X."""

    def __init__(self, madx):
        self._madx = madx
        self._libmadx = libmadx = madx._libmadx
        self._get_element = libmadx.get_global_element
        self._get_element_count = libmadx.get_global_element_count
        self._get_element_index = libmadx.get_global_element_index

    def __iter__(self):
        return iter(map(self._libmadx.get_global_element_name, range(len(self))))

    def __repr__(self):
        return '{{{}}}'.format(', '.join(self))


def cached(func):
    @wraps(func)
    def get(self, *args):
        try:
            val = self._cache[args]
        except KeyError:
            val = self._cache[args] = func(self, *args)
        return val
    return get


class CommandMap(_Mapping):

    def __init__(self, madx):
        self._madx = madx
        self._names = madx._libmadx.get_defined_command_names()
        self._cache = {}

    def __iter__(self):
        return iter(self._names)

    @cached
    def __getitem__(self, name):
        madx = self._madx
        try:
            data = madx._libmadx.get_defined_command(name)
        except ValueError:
            raise KeyError(
                "Not a MAD-X command name: {!r}".format(name)) from None
        return Command(madx, data)

    def __contains__(self, name):
        return name.lower() in self._names

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        return '{{{}}}'.format(', '.join(self))


class BaseTypeMap(CommandMap):

    def __init__(self, madx):
        self._madx = madx
        self._names = madx._libmadx.get_base_type_names()
        self._cache = {}

    @cached
    def __getitem__(self, name):
        return self._madx.elements[name]


class Table(_Mapping):

    """
    MAD-X twiss table.

    Loads individual columns from the MAD-X process lazily only on demand.
    """

    def __init__(self, name, libmadx, *, columns='all', rows='all', _check=True):
        """Just store the table name for now."""
        self._name = name = name.lower()
        self._libmadx = libmadx
        self._columns = columns
        self._rows = rows
        self._cache = {}
        if _check and not libmadx.table_exists(name):
            raise ValueError("Invalid table: {!r}".format(name))

    def selection(self, columns='selected', rows=None) -> "Table":
        """
        Return a Table object that only retrieves the specified rows and
        columns by default.

        :param columns: list of names or 'all' or 'selected'
        :param rows: list of indices or 'all' or 'selected'

        If only ``columns`` is given, ``rows`` defaults to 'selected'
        unless ``columns`` is set to 'all' in which case it also defaults
        to 'all'.
        """
        if rows is None:
            rows = columns if isinstance(columns, str) else 'selected'
        return Table(
            self._name, self._libmadx,
            columns=columns, rows=rows,
            _check=False)

    def __getitem__(self, column):
        """Get the column data."""
        if isinstance(column, int):
            return self.row(column)
        try:
            return self._cache[column.lower()]
        except KeyError:
            return self.reload(column)

    def __iter__(self):
        """Iterate over all column names."""
        return iter(self.col_names())

    def __len__(self):
        """Return number of columns."""
        return self._libmadx.get_table_column_count(self._name, self._columns)

    def __repr__(self):
        return "<{} {!r}: {{{}}}>".format(
            self.__class__.__name__, self._name, ', '.join(self))

    @property
    def summary(self):
        """Get the table summary."""
        return AttrDict(self._libmadx.get_table_summary(self._name))

    def selected_columns(self):
        """Get list of column names that were selected by the user (can be
        empty)."""
        return self._libmadx.get_table_column_names(self._name, selected=True)

    def selected_rows(self):
        """Get list of row indices that were selected by the user (can be
        empty)."""
        return self._libmadx.get_table_selected_rows(self._name)

    def col_names(self, columns=None):
        """Get list of all columns in the table."""
        if columns is None:
            columns = self._columns
        if isinstance(columns, str):
            return self._libmadx.get_table_column_names(
                self._name, selected=columns == 'selected')
        else:
            return columns

    def row_names(self, rows=None):
        """
        Get table row names.

        WARNING: using ``row_names`` after calling ``USE`` (before recomputing
        the table) is unsafe and may lead to segmentation faults or incorrect
        results.
        """
        if rows is None:
            rows = self._rows
        return self._libmadx.get_table_row_names(self._name, rows)

    @property
    def range(self):
        """Get the element names (first, last) of the valid range."""
        row_count = self._libmadx.get_table_row_count(self._name)
        range = (0, row_count-1)
        return tuple(self._libmadx.get_table_row_names(self._name, range))

    def reload(self, column):
        """Reload (recache) one column from MAD-X."""
        try:
            data = self._cache[column.lower()] = self.column(column)
        except ValueError:
            raise KeyError(
                "Unknown table column: {!r}".format(column)) from None
        return data

    def column(self, column: str, rows=None) -> np.ndarray:
        """Retrieve all specified rows in the given column of the table.

        :param column: column name
        :param rows: a list of row indices or ``'all'`` or ``'selected'``
        """
        if rows is None:
            rows = self._rows
        return self._libmadx.get_table_column(self._name, column.lower(), rows)

    def row(self, index, columns=None):
        """Retrieve one row from the table."""
        if columns is None:
            columns = self._columns
        return AttrDict(self._libmadx.get_table_row(self._name, index, columns))

    def copy(self, columns=None, rows=None) -> dict:
        """
        Return a frozen table with the desired columns.

        :param list columns: column names or ``None`` for all columns.
        :returns: column data
        :raises ValueError: if the table name is invalid
        """
        if rows is None:
            rows = columns if isinstance(columns, str) else self._rows
        if rows == self._rows:
            table = self
        else:
            table = self.select(rows=rows)
        return {column: table[column] for column in self.col_names(columns)}

    def dframe(self, columns=None, rows=None, *, index=None):
        """
        Return table as ``pandas.DataFrame``.

        :param columns: column names or 'all' or 'selected'
        :param rows: row indices or 'all' or 'selected'
        :param str index: column name or sequence to be used as index, or
                          ``None`` for using the ``row_names``
        :returns: column data as ``pandas.DataFrame``
        :raises ValueError: if the table name is invalid

        WARNING: using ``index=None`` is unsafe after calling ``USE``.
        In this case, please manually specify another column to be used,
        e.g. ``index="name"``.
        """
        import pandas as pd
        if index is None:
            index = self.row_names(rows)
        elif isinstance(index, str):
            index = util.remove_count_suffix_from_name(self[index])
        else:
            index = index
        return pd.DataFrame(self.copy(columns, rows), index=index)

    def getmat(self, name, idx, *dim):
        s = () if isinstance(idx, int) else (-1,)
        return np.array([
            self[name + ''.join(str(i+1) for i in ijk)][idx]
            for ijk in product(*map(range, dim))
        ]).reshape(dim+s)

    def kvec(self, idx, dim=6):
        """Kicks."""
        return self.getmat('k', idx, dim)

    def rmat(self, idx, dim=6):
        """Sectormap."""
        return self.getmat('r', idx, dim, dim)

    def tmat(self, idx, dim=6):
        """2nd order sectormap."""
        return self.getmat('t', idx, dim, dim, dim)

    def sigmat(self, idx, dim=6):
        """Beam matrix."""
        return self.getmat('sig', idx, dim, dim)


class VarList(_MutableMapping):

    """Mapping of global MAD-X variables."""

    __slots__ = ('_madx', 'cmdpar')

    def __init__(self, madx):
        self._madx = madx
        self.cmdpar = VarParamList(madx)

    def __repr__(self):
        return str({
            key: par.definition
            for key, par in self.cmdpar.items()
            if par.inform
        })

    def __getitem__(self, name):
        return self.cmdpar[name].value

    def __setitem__(self, name, value):
        try:
            var = self.cmdpar[name]
            v, e = var.value, var.expr
        except (TypeError, KeyError):
            v, e = None, None
        if isinstance(value, Number):
            if value != v or e:
                self._madx.input(name + ' = ' + str(value) + ';')
        else:
            if value != e:
                self._madx.input(name + ' := ' + str(value) + ';')

    def __delitem__(self, name):
        raise NotImplementedError("Can't erase a MAD-X global.")

    def __iter__(self):
        """Iterate names of all non-constant globals."""
        return iter(self.cmdpar)

    def __len__(self):
        return len(self.cmdpar)

    @property
    def defs(self):
        return AttrDict({
            key: par.definition
            for key, par in self.cmdpar.items()
        })


class VarParamList(_Mapping):

    """Mapping of global MAD-X variables."""

    __slots__ = ('_madx', '_libmadx')

    def __init__(self, madx):
        self._madx = madx
        self._libmadx = madx._libmadx

    def __getitem__(self, name):
        return self._libmadx.get_var(name.lower())

    def __iter__(self):
        """Iterate names of all non-constant globals."""
        return iter(self._libmadx.get_globals())

    def __len__(self):
        return self._libmadx.num_globals()


class Metadata:

    """MAD-X metadata (license info, etc)."""

    __title__ = u'MAD-X'

    @property
    def __version__(self):
        return self._get_libmadx().get_version_number()

    __summary__ = (
        u'MAD is a project with a long history, aiming to be at the '
        u'forefront of computational physics in the field of particle '
        u'accelerator design and simulation. The MAD scripting language '
        u'is de facto the standard to describe particle accelerators, '
        u'simulate beam dynamics and optimize beam optics.'
    )

    __uri__ = u'http://madx.web.cern.ch/madx/'

    __credits__ = (
        u'MAD-X is developed at CERN and has many contributors. '
        u'For more information see:\n'
        u'\n'
        u'http://madx.web.cern.ch/madx/www/contributors.html'
    )

    def get_copyright_notice(self):
        from cpymad import _read_text
        return _read_text('cpymad.COPYING', 'madx.rst', encoding='utf-8')

    _libmadx = None

    def _get_libmadx(self):
        if not self._libmadx:
            # Need to disable stdin to avoid deadlock that occurs if starting
            # with closed or invalid stdin:
            svc, proc = _rpc.LibMadxClient.spawn_subprocess(stdin=False)
            self._libmadx = svc.libmadx
        return self._libmadx


metadata = Metadata()
