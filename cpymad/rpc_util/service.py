"""
RPC service utilities.
"""

from __future__ import absolute_import

import logging
import traceback
import sys

from . import ipc


__all__ = [
    'Service',
]


class Service(object):

    """
    Base class for a very lightweight synchronous RPC service.

    Counterpart to :class:`Client`.
    """

    def __init__(self, conn):
        """Initialize the service with a :class:`Connection` like object."""
        self._conn = conn

    @classmethod
    def stdio_main(cls, args):
        """Do the full job of preparing and running an RPC service."""
        conn = ipc.prepare_subprocess_ipc(args)
        try:
            svc = cls(conn)
            svc.configure_logging()
            svc.run()
        finally:
            conn.close()

    def configure_logging(self):
        """Configure logging module."""
        logging.basicConfig(logLevel=logging.INFO)

    def run(self):
        """
        Run the service until terminated by either the client or user.

        The service is terminated on user interrupts (Ctrl-C), which might
        or might not be desired.
        """
        while self._communicate():
            pass

    def _communicate(self):
        """
        Receive and serve one RPC request.

        :returns: ``True`` if the service should continue running.
        """
        try:
            request = self._conn.recv()
        except EOFError:
            return False
        except KeyboardInterrupt:
            # Prevent the child process from exiting prematurely if a
            # KeyboardInterrupt (Ctrl-C) is propagated from the parent
            # process. This is important since the parent process might
            # - not exit at all (e.g. interactive python interpretor!) OR
            # - need to perform clean-up that depends on the child process
            return True
        else:
            return self._dispatch(request)

    def _dispatch(self, request):
        """
        Dispatch one RPC request.

        :returns: ``True`` if the service should continue running.
        """
        kind, args = request
        handler = getattr(self, '_dispatch_%s' % (kind,))
        try:
            response = handler(*args)
        except:
            self._reply_exception(sys.exc_info())
        else:
            try:
                self._reply_data(response)
            except ValueError:
                if self._conn.closed:
                    return False
                raise
        return True

    def _dispatch_close(self):
        """Close the connection gracefully as initiated by the client."""
        self._conn.close()

    def _reply_data(self, data):
        """Return data to the client."""
        self._conn.send(('data', (data,)))

    def _reply_exception(self, exc_info):
        """Return an exception state to the client."""
        message = exc_info[0](
            "\n" + "".join(traceback.format_exception(*exc_info))),
        self._conn.send(('exception', message))
