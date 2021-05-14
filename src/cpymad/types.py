"""
Python type analogues for MAD-X data structures.
"""

from collections import namedtuple

__all__ = [
    'Constraint',
    'Parameter',
    'Range',

    'AlignError',
    'FieldError',
    'PhaseError',

    # constants:
    'PARAM_TYPE_LOGICAL',
    'PARAM_TYPE_INTEGER',
    'PARAM_TYPE_DOUBLE',
    'PARAM_TYPE_STRING',
    'PARAM_TYPE_CONSTRAINT',
    'PARAM_TYPE_LOGICAL_ARRAY',
    'PARAM_TYPE_INTEGER_ARRAY',
    'PARAM_TYPE_DOUBLE_ARRAY',
    'PARAM_TYPE_STRING_ARRAY',
    'VAR_TYPE_CONST',
    'VAR_TYPE_DIRECT',
    'VAR_TYPE_DEFERRED',
    'VAR_TYPE_STRING',
]


PARAM_TYPE_LOGICAL       = 0
PARAM_TYPE_INTEGER       = 1
PARAM_TYPE_DOUBLE        = 2
PARAM_TYPE_STRING        = 3
PARAM_TYPE_CONSTRAINT    = 4
PARAM_TYPE_LOGICAL_ARRAY = 10
PARAM_TYPE_INTEGER_ARRAY = 11
PARAM_TYPE_DOUBLE_ARRAY  = 12
PARAM_TYPE_STRING_ARRAY  = 13

VAR_TYPE_CONST    = 0
VAR_TYPE_DIRECT   = 1
VAR_TYPE_DEFERRED = 2
VAR_TYPE_STRING   = 3

dtype_to_native = {
    PARAM_TYPE_LOGICAL: bool,
    PARAM_TYPE_INTEGER: int,
    PARAM_TYPE_DOUBLE: float,
    PARAM_TYPE_STRING: str,
    PARAM_TYPE_LOGICAL_ARRAY: list,
    PARAM_TYPE_INTEGER_ARRAY: list,
    PARAM_TYPE_DOUBLE_ARRAY: list,
    PARAM_TYPE_STRING_ARRAY: list,
}


Range = namedtuple('Range', ['first', 'last'])

AlignError = namedtuple('AlignError', [
    'dx', 'dy', 'ds',
    'dphi', 'dtheta', 'dpsi',
    'mrex', 'mrey', 'mredx', 'mredy',
    'arex', 'arey', 'mscalx', 'mscaly',
])
FieldError = namedtuple('FieldError', ['dkn', 'dks'])
PhaseError = namedtuple('PhaseError', ['dpn', 'dps'])


class Parameter:

    __slots__ = ('name', 'value', 'expr', 'dtype', 'inform', 'var_type')

    def __init__(self, name, value, expr, dtype, inform, var_type=None):
        self.name = name
        self.value = value
        self.expr = expr
        self.dtype = dtype
        self.inform = inform
        if var_type is None:
            if isinstance(value, str):
                var_type = VAR_TYPE_STRING
            else:
                has_expr = expr and (not isinstance(value, list) or any(expr))
                var_type = VAR_TYPE_DEFERRED if has_expr else VAR_TYPE_DIRECT
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


class Constraint:

    """Represents a MAD-X constraint, which has either min/max/both/value."""

    def __init__(self, val=None, min=None, max=None):
        """Just store the values"""
        self.val = val
        self.min = min
        self.max = max
