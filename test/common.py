from cpymad.madx import Madx

from functools import wraps


def with_madx(prompt='X:> ', **madx_kwargs):
    """Decorate a method to be passed a new Madx instance."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with Madx(prompt=prompt, **madx_kwargs) as madx:
                return func(self, madx, *args, **kwargs)
        return wrapper
    return decorator
