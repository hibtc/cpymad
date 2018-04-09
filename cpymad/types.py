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

    def __init__(self, name, value, expr, dtype, inform):
        self.name = name
        self.value = value
        self.expr = expr
        self.dtype = dtype
        self.inform = inform


class Constraint(object):

    """Represents a MAD-X constraint, which has either min/max/both/value."""

    def __init__(self, val=None, min=None, max=None):
        """Just store the values"""
        self.val = val
        self.min = min
        self.max = max
