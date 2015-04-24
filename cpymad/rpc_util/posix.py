"""
Linux specific low level stuff.
"""

from __future__ import absolute_import

import os


__all__ = [
    'Handle',
]


try:
    # python3: handles must be made inheritable explicitly:
    _set_inheritable = os.set_inheritable
except AttributeError:
    # python2: handles are inheritable by default, so nothing to do
    def _set_inheritable(fd, inheritable):
        pass


class Handle(object):

    """
    Wrap a native file descriptor. Close on deletion.

    The API is a compromise aimed at compatibility with win32.
    """

    def __init__(self, handle, own=True):
        """Store a file descriptor (int)."""
        self.handle = handle
        self.own = own

    @classmethod
    def from_fd(cls, fd, own):
        """Create a :class:`Handle` instance from a file descriptor (int)."""
        return cls(fd, own)

    @classmethod
    def pipe(cls):
        """
        Create a unidirectional pipe.

        Return a pair (recv, send) of :class:`Handle`s.
        """
        recv, send = os.pipe()
        return cls(recv), cls(send)

    def __int__(self):
        """Get the underlying file descriptor."""
        return self.handle

    def __del__(self):
        """Close the file descriptor."""
        self.close()

    def __enter__(self):
        """Enter `with` context."""
        return self

    def __exit__(self, *exc_info):
        """Close the file descriptor."""
        self.close()

    def close(self):
        """Close the file descriptor."""
        if self.own and self.handle is not None:
            os.close(self.handle)
            self.handle = None

    def detach_fd(self):
        """Un-own and return the file descriptor."""
        fd, self.handle = self.handle, None
        return fd

    def dup_inheritable(self):
        """Make the file descriptor inheritable."""
        dup = os.dup(self.handle)
        _set_inheritable(dup, True)
        return self.__class__(dup)
