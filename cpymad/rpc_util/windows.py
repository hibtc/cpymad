"""
Windows specific low level stuff.
"""

from __future__ import absolute_import

import sys
import msvcrt

py2 = sys.version_info[0] == 2
if py2:
    import _subprocess as _winapi
    import _multiprocessing
    _CloseHandle = _multiprocessing.win32.CloseHandle
else:
    import _winapi
    _CloseHandle = _winapi.CloseHandle


__all__ = [
    'Handle',
]


if py2:
    # _subprocess.DuplicateHandle and _subprocess.CreatePipe return a
    # handle object similar to the type defined below.
    def _unwrap(handle):
        return handle.Detach()

else:
    # _winapi.DuplicateHandle and _winapi.CreatePipe return plain integers
    # which can be used directly.
    def _unwrap(handle):
        return handle


class Handle(object):

    """
    Wrap a native HANDLE. Close on deletion.
    """

    def __init__(self, handle, own=True):
        """Store a native HANDLE (int)."""
        self.handle = handle
        self.own = own

    @classmethod
    def from_fd(cls, fd, own):
        """Create a :class:`Handle` instance from a file descriptor (int)."""
        handle = msvcrt.get_osfhandle(fd)
        return cls(handle, own)

    @classmethod
    def pipe(cls):
        """
        Create a unidirectional pipe.

        Return a pair (recv, send) of :class:`Handle`s.
        """
        # use _winapi.CreatePipe on windows, just like subprocess.Popen
        # does when requesting PIPE streams. This is the easiest and most
        # reliable method I have tested so far:
        recv, send = _winapi.CreatePipe(None, 0)
        return cls(_unwrap(recv)), cls(_unwrap((send)))

    def __int__(self):
        """Get the underlying handle."""
        return int(self.handle)

    def __del__(self):
        """Close the handle."""
        self.close()

    def __enter__(self):
        """Enter `with` context."""
        return self

    def __exit__(self, *exc_info):
        """Close the handle."""
        self.close()

    def close(self):
        """Close the handle."""
        if self.own and self.handle is not None:
            _CloseHandle(self.handle)
            self.handle = None

    def detach_fd(self):
        """
        Open a file descriptor for the HANDLE and release ownership.

        Closing the file descriptor will also close the handle.
        """
        fd = msvcrt.open_osfhandle(self.handle, 0)
        self.handle = None
        return fd

    def dup_inheritable(self):
        """Point this handle to ."""
        # new handles are created uninheritable by default, but they can be
        # made inheritable on duplication:
        current_process = _winapi.GetCurrentProcess()
        dup = _unwrap(_winapi.DuplicateHandle(
            current_process,                # source process
            self.handle,                    # source handle
            current_process,                # target process
            0,                              # desired access
            True,                           # inheritable
            _winapi.DUPLICATE_SAME_ACCESS,  # options
        ))
        return self.__class__(dup)
