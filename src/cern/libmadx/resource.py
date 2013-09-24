
__all__ = [
    'ResourceProvider', 
    'PackageResource',
    'FileResource',
    'CouchResource'
    ]

import pkg_resources
import json, io
import os

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager, closing
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
import cStringIO


class ResourceProvider(object):
    """
    Abstract base class for resource providers.

    Resources are read-only data objects such as model-data.

    """
    __metaclass__ = ABCMeta

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

        Use this as a context manager only to make sure the file is deleted
        when done using.

        """
        try:
            tempfile = tempfile.NamedTemporaryFile(delete=False)
            with tempfile as dest:
                with self.open(name) as src:
                    copyfileobj(src, dest)
            yield tempfile.name
        finally:
            os.remove(tempfile.name)


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


class FileResource(ResourceProvider):
    """
    File system resource provider.

    Uses the builtins open() to open ordinary files and os.listdir() to list
    directory contents.

    """
    def __init__(self, path):
        self.path = path

    def open(self, name=''):
        return io.open(self._get_path(name))

    def listdir(self, name=''):
        return os.listdir(self._get_path(name))

    def get(self, name=''):
        return self.__class__(self._get_path(name))

    @contextmanager
    def filename(self, name=''):
        yield self._get_path(name)

    def _get_path(self, name):
        if not name:
            return self.path
        elif isinstance(name, list):
            return os.path.join(self.path, *name)
        else:
            return os.path.join(self.path, name)

    def provider(self):
        return self.__class__(os.path.dirname(self.path))


class CouchResource(ResourceProvider):
    """
    CouchDB document resource provider.

    Uses a couchdb.Database instance to retrieve couchdb documents and
    attachments.

    """
    def __init__(self, db, doc=None, file=None):
        self.db = db
        self.doc = doc
        self.file = file

    def open(self, name=''):
        if name:
            return self.get(name).open()
        if self.file:
            return self.db.get_attachment(self.doc, name)
        elif self.doc:
            # TODO: deal with unicode
            return io.StringIO(self.load())
        else:
            raise NotImplementedError("Database is not a loadabe resource.")

    def load(self, name=''):
        if name:
            return self.get(name).load()
        if self.file:
            return super(CouchResource, self).load()
        elif self.doc:
            return self.db[self.doc]
        else:
            raise NotImplementedError("Database is not a loadabe resource.")

    def listdir(self, name=''):
        if name:
            return self.get(name).listdir()
        if self.file:
            raise NotImplementedError("Cannot recurse into attachment.")
        elif self.doc:
            return (fname for fname in self.db[self.doc]['_attachments'])
        else:
            return (docname for docname in self.db)

    def get(self, name=''):
        if self.file:
            return CouchResource(self.db, self.doc, os.path.join(self.name, name))
        elif self.doc:
            return CouchResource(self.db, self.doc, name)
        else:
            return CouchResource(self.db, name)

    def provider(self):
        if self.file:
            return CouchResource(self.db, self.doc)
        else:
            return CouchResource(self.db)

    def listdir_filter(self, name='', ext=''):
        return self.listdir(name)

