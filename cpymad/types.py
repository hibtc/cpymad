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

    __slots__ = ('name', 'value', 'expr', 'dtype', 'inform', 'var_type')

    def __init__(self, name, value, expr, dtype, inform, var_type=None):
        self.name = name
        self.value = value
        self.expr = expr
        self.dtype = dtype
        self.inform = inform
        if var_type is None:
            if isinstance(value, str):
                var_type = 3
            elif isinstance(value, list):
                var_type = 2 if expr and any(expr) else 1
            else:
                var_type = 2 if expr else 1
        self.var_type = var_type

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
