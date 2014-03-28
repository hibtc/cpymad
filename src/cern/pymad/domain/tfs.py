#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
"""
Created on 16 Aug 2011
.. module:: tfs
.. moduleauthor:: kfuchsbe
"""


class LookupDict(object):

    """
    An attribute access view for a dictionary.

    The dictionary entries are accessible both via attribute and key access.
    Attribute (and key) access is always case insensitive.
    """

    def __init__(self, data):
        """
        Initialize the object with a copy of the data dictionary.

        :param dict data: original data
        """
        # store the data in a new dict, to unify the keys to lowercase
        self._data = dict()
        for key, val in data.items():
            self._data[self._unify_key(key)] = val

    def __getstate__(self):
        """Serialize to a primitive dict (pickle)."""
        return self._data

    def __setstate__(self, state):
        """Deserialize from a primitive dict (unpickle)."""
        self._data = state

    def __iter__(self):
        """Return iterator over the (lowercase) keys."""
        return iter(self._data)

    def __getattr__(self, name):
        """
        Return value associated to the given key.

        :param str name: case-insensitive key
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, key):
        """
        Return value associated to the given key.

        :param str key: case-insensitive key
        """
        return self._data[self._unify_key(key)]

    def _unify_key(self, key):
        """
        Convert key to lowercase.

        :param str key: case-insensitive key
        """
        return key.lower()

    def keys(self):
        """
        Return iterable over all keys in the dictionary.
        """
        return self._data.keys()


class TfsTable(LookupDict):

    """Result table of a TWISS calculation with case-insensitive keys."""

    def __init__(self, data):
        """
        Initialize the object with a copy of the data dictionary.

        :param dict data: original data

        If not present in the original data, the key 'names' is aliased to
        'name', whose value contains a list of all element node names.
        """
        LookupDict.__init__(self, data)
        try:
            self._data.setdefault('names', self._data['name'])
        except KeyError:
            pass


class TfsSummary(LookupDict):

    """Summary table of a TWISS with case-insensitive keys."""

    pass

