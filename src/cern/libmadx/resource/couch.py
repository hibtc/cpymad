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
        """
        Initialize a couchdb resource provider.

        The parameters are:

        :param object db: a couchdb Database instance
        :param string doc: name (id) of the selected document (may be empty)
        :param string file: name of the selected attachment (may be empty)

        If doc and file are empty, a selection (.get) will yield a
        document. Otherwise a selection will append to the name of the
        currently selected attachment.

        """
        self.db = db
        self.doc = doc
        self.file = file
        assert self.doc or not self.file

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
            return (fname for fname in self.db[self.doc]['_attachments']
                    if fname.startswith(self.file))
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

