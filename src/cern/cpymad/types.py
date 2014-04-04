"""
Python type analogues for MAD-X data structures.
"""

from collections import namedtuple

__all__ = ['LookupDict',
           'TfsTable',
           'TfsSummary',
           'Range',
           'Constraint',
           'Expression',
           'Element']


class LookupDict(object):

    """
    An attribute access view for a dictionary.

    The dictionary entries are accessible both via attribute and key access.
    Attribute (and key) access is always case insensitive.
    """

    def __init__(self, data):
        """
        Initialize the object with a copy of the data dictionary.

        :param dict data: original data
        """
        # store the data in a new dict, to unify the keys to lowercase
        self._data = dict()
        for key, val in data.items():
            self._data[self._unify_key(key)] = val

    def __getstate__(self):
        """Serialize to a primitive dict (pickle)."""
        return self._data

    def __setstate__(self, state):
        """Deserialize from a primitive dict (unpickle)."""
        self._data = state

    def __iter__(self):
        """Return iterator over the (lowercase) keys."""
        return iter(self._data)

    def __getattr__(self, name):
        """
        Return value associated to the given key.

        :param str name: case-insensitive key
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, key):
        """
        Return value associated to the given key.

        :param str key: case-insensitive key
        """
        return self._data[self._unify_key(key)]

    def _unify_key(self, key):
        """
        Convert key to lowercase.

        :param str key: case-insensitive key
        """
        return key.lower()

    def keys(self):
        """
        Return iterable over all keys in the dictionary.
        """
        return self._data.keys()


class TfsTable(LookupDict):

    """Result table of a TWISS calculation with case-insensitive keys."""

    def __init__(self, data):
        """
        Initialize the object with a copy of the data dictionary.

        :param dict data: original data

        If not present in the original data, the key 'names' is aliased to
        'name', whose value contains a list of all element node names.
        """
        LookupDict.__init__(self, data)
        try:
            self._data.setdefault('names', self._data['name'])
        except KeyError:
            pass


class TfsSummary(LookupDict):

    """Summary table of a TWISS with case-insensitive keys."""

    pass


Range = namedtuple('Range', ['first', 'last'])


class Constraint(object):

    """Represents a MAD-X constraint, which has either min/max/both/value."""

    def __init__(self, val=None, min=None, max=None):
        """Just store the values"""
        self.val = val
        self.min = min
        self.max = max


class Expression(object):

    """
    Data structure representing input values from madx statements.

    These both an expression (str) and a value (bool/int/float).
    """

    def __init__(self, expr, value, type=float):
        """Store string expression and value."""
        self.expr = expr
        self._value = value
        self.type = type

    def __repr__(self):
        """Return string representation of this object."""
        return '{}({!r}, {}, {})'.format(self.__class__.__name__,
                                         self.expr, self.value,
                                         self.type.__name__)

    def __str__(self):
        """Get the expression as string."""
        return self.expr

    @property
    def value(self):
        """Get the value with the most accurate type."""
        return self.type(self._value)

    def __bool__(self):     # python3
        """Get the value as boolean."""
        return bool(self._value)

    __nonzero__ = __bool__   # python2

    def __int__(self):
        """Get the value as integer."""
        return int(self._value)

    def __float__(self):
        """Get the value as double."""
        return float(self._value)


class Element(LookupDict):

    """
    Case-insensitive property table for a MAD-X beamline element.
    """

    pass
