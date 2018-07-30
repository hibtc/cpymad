"""
Classes that represent various types of MAD-X objects:

- commands (:class:`Command`)
- elements (:class:`Element`)
- tables (:class:`Table`)
- lists of commands/elements/variables/tables
"""

from functools import wraps
from itertools import product
from numbers import Number
import collections

import numpy as np

from . import util

try:
    basestring
except NameError:
    basestring = str


class _Mapping(collections.Mapping):

    def __repr__(self):
        """String representation of a custom mapping object."""
        return str(dict(self))

    def __str__(self):
        return repr(self)

    def __getattr__(self, key):
        key = _fix_name(key)
        try:
            return self[key]
        except KeyError:
            return self._missing(key)

    def _missing(self, key):
        raise AttributeError(key)


class _MutableMapping(_Mapping, collections.MutableMapping):

    __slots__ = ()

    def __setattr__(self, key, val):
        if key in self.__slots__:
            object.__setattr__(self, key, val)
        else:
            key = _fix_name(key)
            self[key] = val

    def __delattr__(self, key):
        if key in self.__slots__:
            object.__delattr__(self, key)
        else:
            key = _fix_name(key)
            del self[key]


class AttrDict(_Mapping):

    def __init__(self, data):
        if not isinstance(data, collections.Mapping):
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


class Sequence(object):

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

    # in py3 __ne__ delegates to __eq__, but we still need this for py2:
    def __ne__(self, other):
        """Comparison by sequence name."""
        return not (self == other)

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
            raise KeyError

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
            raise KeyError

    def __contains__(self, name):
        return self._libmadx.table_exists(name.lower())

    def __len__(self):
        return self._libmadx.get_table_count()


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
        return self._data[name.lower()].value

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
        self._madx.input(util.format_command(self, *args, **kwargs))

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
        raise AttributeError('Unknown attribute {!r} for {!r} command!'
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


class ArrayAttribute(collections.Sequence):

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

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return str(self._values)

    def __str__(self):
        return str(self._values)


class BaseElementList(object):

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
        if isinstance(index, basestring):
            # allow element names to be passed for convenience:
            try:
                index = self.index(index)
            except ValueError:
                raise KeyError("Unknown element: {!r}".format(index))
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


class ElementList(BaseElementList, collections.Sequence):

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
        return self._libmadx.get_expanded_element(self._sequence_name, element_index)

    def _get_element_count(self):
        return self._libmadx.get_expanded_element_count(self._sequence_name)

    def _get_element_index(self, element_name):
        return self._libmadx.get_expanded_element_index(self._sequence_name, element_name)

    def _get_element_at(self, pos):
        return self._libmadx.get_expanded_element_index_by_position(self._sequence_name, pos)

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
            raise KeyError("Not a MAD-X command name: {!r}".format(name))
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

    def __init__(self, name, libmadx, _check=True):
        """Just store the table name for now."""
        self._name = name = name.lower()
        self._libmadx = libmadx
        self._cache = {}
        if _check and not libmadx.table_exists(name):
            raise ValueError("Invalid table: {!r}".format(name))

    def __getitem__(self, column):
        """Get the column data."""
        if isinstance(column, int):
            return self.row(column)
        try:
            return self._cache[column.lower()]
        except KeyError:
            return self.reload(column)

    def _query(self, column):
        """Retrieve the column data."""
        try:
            return self._libmadx.get_table_column(self._name, column.lower())
        except ValueError:
            raise KeyError(column)

    def __iter__(self):
        """Iterate over all column names."""
        return iter(self._libmadx.get_table_column_names(self._name) or
                    self._libmadx.get_table_column_names_all(self._name))

    def __len__(self):
        """Return number of columns."""
        return (self._libmadx.get_table_column_count(self._name) or
                self._libmadx.get_table_column_count_all(self._name))

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name)

    @property
    def summary(self):
        """Get the table summary."""
        return AttrDict(self._libmadx.get_table_summary(self._name))

    @property
    def range(self):
        """Get the element names (first, last) of the valid range."""
        row_count = self._libmadx.get_table_row_count(self._name)
        range = (0, row_count-1)
        return tuple(self._libmadx.get_table_row_names(self._name, range))

    def reload(self, column):
        """Reload (recache) one column from MAD-X."""
        self._cache[column.lower()] = data = self._query(column)
        return data

    def row(self, index, columns='selected'):
        """Retrieve one row from the table."""
        return AttrDict(self._libmadx.get_table_row(self._name, index, columns))

    def copy(self, columns=None):
        """
        Return a frozen table with the desired columns.

        :param list columns: column names or ``None`` for all columns.
        :returns: column data
        :rtype: dict
        :raises ValueError: if the table name is invalid
        """
        if columns is None:
            columns = self
        return {column: self[column] for column in columns}

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


def _fix_name(name):
    if name.startswith('_'):
        raise AttributeError("Unknown item: {!r}! Did you mean {!r}?"
                             .format(name, name.strip('_') + '_'))
    if name.endswith('_'):
        name = name[:-1]
    return name
