#----------------------------------------
# package.py by Thomas Gläßle
# 
# To the extent possible under law, the person who associated CC0 with
# package.py has waived all copyright and related or neighboring rights
# to package.py.
# 
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
#----------------------------------------
"""
Resource provider for python package resources.
"""
__all__ = ['PackageResource']

import pkg_resources
from contextlib import contextmanager, closing
from shutil import rmtree
from os import remove
from os.path import isdir

from .base import ResourceProvider


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
        filename = pkg_resources.resource_filename(
                self.package,
                self._get_path(name))
        # This check is not very forward compatible, but its the best I got:
        extracted = isinstance(pkg_resources.get_provider(self.package),
                               pkg_resources.ZipProvider)
        try:
            yield filename
        finally:
            if extracted:
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
            return '/'.join([self.path, name])

    def provider(self):
        parts = self.path.rsplit('/', 1)
        return self.__class__(self.package, parts[0] if len(parts) > 1 else '')

