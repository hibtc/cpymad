"""
Python type analogues for MAD-X data structures.
"""

from collections import namedtuple

__all__ = [
    'Range',
    'Parameter',
    'Constraint',
]


Range = namedtuple('Range', ['first', 'last'])

class Parameter(object):

    __slots__ = ('name', 'value', 'expr', 'dtype', 'inform')

    def __init__(self, name, value, expr, dtype, inform):
        self.name = name
        self.value = value
        self.expr = expr
        self.dtype = dtype
        self.inform = inform

    def __call__(self):
        return self.definition

    @property
    def definition(self):
        """Return command argument as should be used for MAD-X input to
        create an identical element."""
        if isinstance(self.value, list):
            return [e or v for v, e in zip(self.value, self.expr)]
        else:
            return self.expr or self.value

    def __str__(self):
        return str(self.definition)


class Constraint(object):

    """Represents a MAD-X constraint, which has either min/max/both/value."""

    def __init__(self, val=None, min=None, max=None):
        """Just store the values"""
        self.val = val
        self.min = min
        self.max = max
