from cpymad.madx import Madx, CommandLog

import sys
from functools import wraps


def create_madx(**kwargs):
    kwargs.setdefault('command_log', CommandLog(sys.stdout, 'X:> '))
    return Madx(**kwargs)


def with_madx(**madx_kwargs):
    """Decorate a method to be passed a new Madx instance."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with create_madx(**madx_kwargs) as madx:
                return func(self, madx, *args, **kwargs)
        return wrapper
    return decorator
