"""
Monitor file objects.

Python2 module only.
"""

from __future__ import absolute_import

import __builtin__
import weakref


__all__ = [
    'File',
    'monkey_patch',
    'unpatch',
]


builtin_file = __builtin__.file
builtin_open = __builtin__.open


class File(builtin_file):

    __doc__ = builtin_file.__doc__
    __doc__ += """

-----------------------------------------------------------------
This class is a light wrapper for the builtin `file` type.  It
keeps track of all its objects in the classvariable `_instances`.
This can be useful to close all files e.g. when spawning a
subprocess.
-----------------------------------------------------------------
"""

    _instances = weakref.WeakSet()

    def __init__(self, *args, **kwargs):
        super(File, self).__init__(*args, **kwargs)
        self._instances.add(self)

    __init__.__doc__ = open.__doc__


def monkey_patch():
    """Replace file and open builtins."""
    __builtin__.file = File
    __builtin__.open = File


def unpatch():
    """Restore file and open builtins."""
    __builtin__.file = builtin_file
    __builtin__.open = builtin_open
