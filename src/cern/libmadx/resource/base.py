"""
Base class for resource providers.
"""

__all__ = ['ResourceProvider']

import json
import os

from contextlib import contextmanager
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from cern.pymad.abc.interface import Interface, abstractmethod


class ResourceProvider(Interface):
    """
    Abstract base class for resource providers.

    Resources are read-only data objects such as model-data.

    """
    @abstractmethod
    def open(self, name=''):
        """
        Open the specified resource.

        Returns a file-like object. When finished using close() must be
        called on the returned object.

        """
        pass

    @abstractmethod
    def listdir(self, name=''):
        """
        List directory contents.

        This works similar to os.listdir().
        restrict can be used to filter for certain file types.

        """
        pass

    @abstractmethod
    def get(self, name):
        """
        Get a provider object relative to the specified subdirectory.

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

        NOTE: To stay upward compatible ext should only be passed as keyword
        argument.

        """
        for res_name in self.listdir(name):
            if res_name.lower().endswith(ext):
                yield res_name

    def load(self, name=''):
        """
        Load the specified resource into memory and return all the data.

        This is a convenience mixin.

        """
        with self.open(name) as f:
            return f.read()

    def json(self, name='', **kwargs):
        """
        Load the specified json resource.

        kwargs can be used to pass additional arguments to the json parser.
        This is a convenience mixin.

        """
        with self.open(name) as f:
            return json.load(f)

    @contextmanager
    def filename(self, name=''):
        """
        Yield the path of a file containing the resource.

        Use this as a context manager to make sure the file is deleted when
        done using.

        """
        try:
            tempfile = tempfile.NamedTemporaryFile(delete=False)
            with tempfile as dest:
                with self.open(name) as src:
                    copyfileobj(src, dest)
            yield tempfile.name
        finally:
            os.remove(tempfile.name)

