"""
RPC client utilities.
"""

from __future__ import absolute_import

import sys

from . import ipc


__all__ = [
    'RemoteProcessClosed',
    'RemoteProcessCrashed',
    'Client',
]


class RemoteProcessClosed(RuntimeError):
    """The MAD-X remote process has already been closed."""
    pass


class RemoteProcessCrashed(RuntimeError):
    """The MAD-X remote process has crashed."""
    pass


class Client(object):

    """
    Base class for a very lightweight synchronous RPC client.

    Uses a connection that shares the interface with :class:`Connection` to
    do synchronous RPC. Synchronous IO means that currently callbacks /
    events are impossible.
    """

    def __init__(self, conn):
        """Initialize the client with a :class:`Connection` like object."""
        self._conn = conn

    def __del__(self):
        """Close the client and the associated connection with it."""
        try:
            self.close()
        except (RemoteProcessCrashed, RemoteProcessClosed):
            # catch ugly follow-up warnings after a MAD-X process has crashed
            pass

    @classmethod
    def spawn_subprocess(cls, **Popen_args):
        """
        Create client for a backend service in a subprocess.

        You can use the keyword arguments to pass further arguments to
        Popen, which is useful for example, if you want to redirect STDIO
        streams.
        """
        args = [sys.executable, '-m', cls.__module__]
        conn, proc = ipc.spawn_subprocess(args, **Popen_args)
        return cls(conn), proc

    def close(self):
        """Close the connection gracefully, stop the remote service."""
        try:
            self._conn.send(('close', ()))
        except ValueError:      # already closed
            pass
        self._conn.close()

    @property
    def closed(self):
        """Check if connection is closed."""
        return self._conn.closed

    def _request(self, kind, *args):
        """Communicate with the remote service synchronously."""
        try:
            self._conn.send((kind, args))
        except ValueError:
            if self.closed:
                raise RemoteProcessClosed()
            raise
        except IOError:
            raise RemoteProcessCrashed()
        try:
            response = self._conn.recv()
        except EOFError:
            raise RemoteProcessCrashed()
        return self._dispatch(response)

    def _dispatch(self, response):
        """Dispatch an answer from the remote service."""
        kind, args = response
        handler = getattr(self, '_dispatch_%s' % (kind,))
        return handler(*args)

    def _dispatch_exception(self, exc_info):
        """Dispatch an exception."""
        raise exc_info

    def _dispatch_data(self, data):
        """Dispatch returned data."""
        return data
