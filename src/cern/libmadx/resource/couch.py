"""
Resource provider for couchdb resources.
"""

__all__ = ['CouchResource']

from io import StringIO
import os.path

from .base import ResourceProvider


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
            return StringIO(self.load())
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

