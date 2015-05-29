# encoding: utf-8
"""
Contains base class for resource providers.
"""

import csv
import os
import sys
from contextlib import contextmanager
from shutil import copyfileobj
from tempfile import NamedTemporaryFile


__all__ = [
    'ResourceProvider',
]


class ResourceProvider(object):

    """
    Abstract base class for resource providers.

    Resources are read-only (at the moment) data objects such as
    model-data. Resource providers have a common interface to abstract the
    underlying API to access those resources.
    """

    def open(self, name='', encoding=None):
        """
        Open the specified resource.

        :param string name: Name of the resource, optional.
        :param string encoding: Either None or encoding to use (optional).

        Returns a file-like object. When finished using ``close()`` must be
        called on the returned object. If ``encoding`` is ``None`` the
        stream is opened in binary mode.
        """
        raise NotImplementedError("ResourceProvider.open")

    def listdir(self, name=''):
        """
        List directory contents.

        :param string name: Name of the resource, optional.

        This works similar to os.listdir().
        restrict can be used to filter for certain file types.
        """
        raise NotImplementedError("ResourceProvider.listdir")

    def get(self, name):
        """
        Get a provider object relative to the specified subdirectory.

        :param string name: Name of the resource, optional.

        Returns an instance of the ResourceProvider that opens, lists and
        gets objects relative to the specified subdirectory.
        """
        raise NotImplementedError("ResourceProvider.get")

    def provider(self):
        """
        Get the parent provider object.
        """
        raise NotImplementedError("ResourceProvider.provider")

    # mixins:
    def listdir_filter(self, name='', ext=''):
        """
        List resources that match restriction.

        :param string name: Name of the resource, optional.
        :param string ext: Filename extension to be filtered (including dot).

        NOTE: To stay upward compatible ext should only be passed as keyword
        argument.
        """
        for res_name in self.listdir(name):
            if res_name.lower().endswith(ext):
                yield res_name

    def load(self, name='', encoding=None):
        """
        Load the specified resource into memory and return all the data.

        :param string name: Name of the resource, optional.
        :param string encoding: Either None or encoding to use (optional).

        If ``encoding`` is ``None`` the returned data is binary.
        This is a convenience mixin.
        """
        with self.open(name, encoding) as f:
            return f.read()

    def yaml(self, name='', encoding='utf-8'):
        """
        Load the specified YAML/JSON resource.

        :param string name: Name of the resource, optional.
        :param string encoding: Encoding to use.

        kwargs can be used to pass additional arguments to the YAML parser.
        This is a convenience mixin.

        Note that a YAML parser is used but since JSON is a subset of YAML
        this function can also be used to load JSON resources. The input is
        not checked to be valid JSON!
        """
        import yaml
        try:
            Loader = yaml.CSafeLoader
        except AttributeError:
            Loader = yaml.SafeLoader
        with self.open(name, encoding=encoding) as f:
            return yaml.load(f, Loader)

    # backward compatibility alias
    json = yaml

    if sys.version_info[0] == 2:
        def csv(self, filename, encoding='utf-8', **kwargs):
            """Load unicode CSV file, return iterable over rows."""
            with self.open(filename) as f:
                for row in csv.reader(f, **kwargs):
                    yield [e.decode(encoding) for e in row]

    else:
        def csv(filename, encoding='utf-8', **kwargs):
            """Load unicode CSV file, return iterable over rows."""
            with self.open(filename, encoding=encoding) as f:
                return csv.reader(list(f), **kwargs)

    @contextmanager
    def filename(self, name=''):
        """
        Yield the path of a file containing the resource.

        Use this as a context manager to make sure temporary files are
        deleted when done using. Example:

        .. code-block:: python

            res = PackageResource('foo_module_123') # may .egg or filesystem
            with res.filename('foo.txt') as filename:
                with open(filename) as file:
                    content = file.read()
            # temporarily extracted files are deleted at this point
        """
        try:
            tempfile = tempfile.NamedTemporaryFile(mode='wb', delete=False)
            with tempfile as dest:
                with self.open(name) as src:
                    copyfileobj(src, dest)
            yield tempfile.name
        finally:
            os.remove(tempfile.name)
