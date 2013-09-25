"""
Resource provider for python package resources.
"""

__all__ = ['PackageResource']

import pkg_resources
from contextlib import contextmanager, closing

from .base import ResourceProvider


class PackageResource(ResourceProvider):
    """
    Provider for resources that are inside a python package.

    Uses pkg_resources.resource_stream() to open resources and
    pkg_resources.listdir() to list available resources.

    """
    def __init__(self, package, path=''):
        self.package = package
        self.path = path

    def open(self, name=''):
        stream = pkg_resources.resource_stream(
                self.package,
                self._get_path(name))
        if not (hasattr(stream, '__enter__') and hasattr(stream, '__exit__')):
            stream = closing(stream)
        return stream

    def load(self, name=''):
        return pkg_resources.resource_string(
                self.package,
                self._get_path(name))

    def listdir(self, name=''):
        return pkg_resources.resource_listdir(
                self.package,
                self._get_path(name))

    def get(self, name=''):
        return self.__class__(
                self.package,
                self._get_path(name))

    @contextmanager
    def filename(self, name=''):
        yield pkg_resources.resource_filename(
                self.package,
                self._get_path(name))
        # TODO: cleanup
        # there is pkg_resources.cleanup_resources but this deletes all
        # cached resources.

    def _get_path(self, name):
        if not name:
            return self.path
        elif isinstance(name, list):
            return '/'.join([self.path] + name)
        else:
            return '/'.join([self.path, name])

    def provider(self):
        parts = self.path.rsplit('/', 1)
        return self.__class__(self.package, parts[0] if len(parts) > 1 else '')

