# encoding: utf-8
#----------------------------------------
# base.py by Thomas Gläßle
# 
# To the extent possible under law, the person who associated CC0 with
# base.py has waived all copyright and related or neighboring rights
# to base.py.
# 
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
#----------------------------------------
"""
Contains base class for resource providers.
"""
__all__ = ['ResourceProvider']

import yaml
import os

from contextlib import contextmanager
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from cern.pymad.abc.interface import Interface, abstractmethod


class ResourceProvider(Interface):
    """
    Abstract base class for resource providers.

    Resources are read-only (at the moment) data objects such as
    model-data. Resource providers have a common interface to abstract the
    underlying API to access those resources.

    """
    @abstractmethod
    def open(self, name='', encoding=None):
        """
        Open the specified resource.

        :param string name: Name of the resource, optional.
        :param string encoding: Either None or encoding to use (optional).

        Returns a file-like object. When finished using ``close()`` must be
        called on the returned object. If ``encoding`` is ``None`` the
        stream is opened in binary mode.

        """
        pass

    @abstractmethod
    def listdir(self, name=''):
        """
        List directory contents.

        :param string name: Name of the resource, optional.

        This works similar to os.listdir().
        restrict can be used to filter for certain file types.

        """
        pass

    @abstractmethod
    def get(self, name):
        """
        Get a provider object relative to the specified subdirectory.

        :param string name: Name of the resource, optional.

        Returns an instance of the ResourceProvider that opens, lists and
        gets objects relative to the specified subdirectory.

        """
        pass

    @abstractmethod
    def provider(self):
        """
        Get the parent provider object.
        """
        pass

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

    def yaml(self, name='', encoding='utf-8', **kwargs):
        """
        Load the specified yaml resource.

        :param string name: Name of the resource, optional.
        :param string encoding: Encoding to use.

        kwargs can be used to pass additional arguments to the yaml parser.
        This is a convenience mixin.

        """
        with self.open(name, encoding=encoding) as f:
            return yaml.load(f, **kwargs)

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

