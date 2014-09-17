"""
Python type analogues for MAD-X data structures.
"""

from collections import namedtuple

__all__ = [
    'Range',
    'Constraint',
    'Expression',
]


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
