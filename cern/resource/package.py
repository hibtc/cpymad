# encoding: utf-8
"""
Resource provider for python package resources.
"""

import pkg_resources
from contextlib import contextmanager, closing
from shutil import rmtree
from os import remove
from os.path import isdir
from io import StringIO, BytesIO, open

from .base import ResourceProvider


__all__ = [
    'PackageResource',
]


_manager = pkg_resources.ResourceManager()


class PackageResource(ResourceProvider):

    """
    Provider for resources that are inside a python package.

    This can be used to access data that resides within .egg files as well
    as data that is accessible through the filesystem.

    Uses pkg_resources.resource_stream() to open resources and
    pkg_resources.listdir() to list available resources.
    """

    def __init__(self, package, path=''):
        """
        Initialize package resource provider.

        :param string package: python package/module name or object
        :param string path: name of a resource relative to the package
        """
        self.package = package
        self.path = path
        self._manager = _manager
        self._provider = pkg_resources.get_provider(package)

    def open(self, name='', encoding=None):
        # The python2 implementation of ZipProvider.get_resource_stream
        # uses cStringIO.StringIO which is unacceptable for our purposes.
        # It is not usable as context manager, it fails to support unicode,
        # it cannot be wrapped by TextIOWrapper. This is how it should have
        # been done from the beginning:
        if self._is_filesystem:
            filename = self._provider.get_resource_filename(
                self._manager,
                self._get_path(name))
            if encoding:
                return open(filename, 'rt', encoding=encoding)
            else:
                return open(filename, 'rb')
        else:
            data = self.load(name, encoding)
            if encoding:
                return StringIO(data)
            else:
                return BytesIO(data)

    def load(self, name='', encoding=None):
        data = self._provider.get_resource_string(
                self._manager,
                self._get_path(name))
        if encoding:
            return data.decode(encoding)
        else:
            return data

    def listdir(self, name=''):
        return self._provider.resource_listdir(self._get_path(name))

    def get(self, name=''):
        return self.__class__(self.package, self._get_path(name))

    @contextmanager
    def filename(self, name=''):
        filename = self._provider.get_resource_filename(
            self._manager,
            self._get_path(name))
        try:
            yield filename
        finally:
            if self._is_extracted:
                # there is also pkg_resources.cleanup_resources but this
                # deletes all cached resources. Furthermore, that method
                # seems to be a NOOP, currently.
                if isdir(filename):
                    rmtree(filename)
                else:
                    remove(filename)


    def _get_path(self, name):
        if not name:
            return self.path
        elif isinstance(name, list):
            return '/'.join([self.path] + name)
        else:
            return '/'.join([self.path] + name.split('/'))

    @property
    def _is_filesystem(self):
        return isinstance(self._provider, pkg_resources.DefaultProvider)

    @property
    def _is_extracted(self):
        # This check is not very forward compatible, but its the best I got:
        return isinstance(self._provider, pkg_resources.ZipProvider)

    def provider(self):
        parts = self.path.rsplit('/', 1)
        return self.__class__(self.package, parts[0] if len(parts) > 1 else '')
