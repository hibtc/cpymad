"""
RPC service module for libmadx.

This module is needed to execute several instances of the `libmadx` module
in remote processes and communicate with them via remote procedure calls
(RPC). Use the :meth:`LibMadxClient.spawn` to create a new instance.


CAUTION:

- the service communicates with the remote end via pickling, i.e. both
  ends can execute arbitrary code on the other side. This means that the
  remote process can not be used to safely execute unsafe commands.

- When launching the remote process you should make sure on your own that
  the process does not inherit any system resources. On python>=2.7 all
  handles will be closed automatically when doing `execv`, i.e. when using
  `subprocess.Popen` - but NOT with `multiprocessing.Process`. I don't know
  any reliable method that works on python2.6. For more information, see:

  http://www.python.org/dev/peps/pep-0446/

"""
from __future__ import absolute_import

__all__ = ['LibMadxClient']

import traceback
import os
import sys
if sys.platform == 'linux':
    from ._connection.multiprocessing import Connection
else:
    from ._connection.pickle import Connection

def remap_stdio():
    """
    Remap STDIO streams to new file descriptors and create new STDIO streams.

    Create new file descriptors for the original STDIO streams. Then
    replace the python STDIO file objects with newly opened streams:

    :obj:`sys.stdin` is mapped to a NULL stream.
    :obj:`sys.stdout` is initialized with the current console.

    :returns: the remapped (STDIN, STDOUT) file descriptors

    """
    # This function can only make sure that the original file descriptors
    # of sys.stdin, sys.stdout, sys.stderr are remapped correctly. It can
    # make no guarantees about the standard POSIX file descriptors (0, 1).
    # Usually though, these should be the same.
    STDIN = sys.stdin.fileno()
    STDOUT = sys.stdout.fileno()
    # virtual file name for console (terminal) IO:
    console = 'con:' if sys.platform == 'win32' else '/dev/tty'
    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        stdout_fd = os.open(console, os.O_WRONLY)
    except (IOError, OSError):
        stdout_fd = os.open(os.devnull, os.O_WRONLY)
    # The original stdio streams can only be closed *after* opening new
    # stdio streams to avoid the risk that the file descriptors will be
    # reused immediately. But before closing, their file descriptors need
    # to be duplicated:
    recv_fd = os.dup(sys.stdin.fileno())
    send_fd = os.dup(sys.stdout.fileno())
    sys.stdin.close()
    sys.stdout.close()
    # By duplicating the file descriptors to the STDIN/STDOUT file
    # descriptors non-python libraries can make use of these streams as
    # well. The initial fds are not needed anymore.
    os.dup2(stdin_fd, STDIN)
    os.dup2(stdout_fd, STDOUT)
    os.close(stdin_fd)
    os.close(stdout_fd)
    # Create new python file objects for STDIN/STDOUT and remap the
    # corresponding file descriptors: Reopen python standard streams. This
    # enables all python modules to use these streams. Note: the stdout
    # buffer length is set to '1', making it line buffered, which behaves
    # like the default in most circumstances.
    sys.stdin = os.fdopen(STDIN, 'rt')
    sys.stdout = os.fdopen(STDOUT, 'wt', 1)
    # Return the remapped file descriptors of the original STDIO streams
    return recv_fd, send_fd

# Client side code:
class Client(object):
    """
    Base class for a very lightweight generic RPC client.

    Uses a connection that shares the interface with :class:`Connection` to
    do synchronous RPC. Synchronous IO means that currently callbacks /
    events are impossible.

    """
    def __init__(self, conn):
        """Initialize the client with a :class:`Connection` like object."""
        self._conn = conn

    def __del__(self):
        """Close the client and the associated connection with it."""
        self.close()

    @classmethod
    def spawn_subprocess(cls, entry=__name__):
        """
        Create client for a backend service in a subprocess.

        :param str entry: module name for the remote entry point
        :returns: client object for the spawned subprocess.

        The ``entry`` parameter determines which module to execute in the
        remote process. The '__main__' code branch in that module should
        execute :meth:`Service.stdio_main` for the corresponding service.

        Inter-process communication (IPC) with the remote process is
        performed via its STDIO streams.

        """
        args = [sys.executable, '-u', '-m', entry]
        conn = Connection.to_subprocess(args)
        return cls(conn)

    def close(self):
        """Close the connection gracefully, stop the remote service."""
        try:
            self._conn.send(('close', ()))
        except ValueError:      # already closed
            pass
        self._conn.close()

    def _request(self, kind, *args):
        """Communicate with the remote service synchronously."""
        self._conn.send((kind, args))
        return self._dispatch(self._conn.recv())

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

class Service(object):
    """
    Base class for a very lightweight generic RPC service.

    This is the counterpart to :class:`Client`.

    """
    def __init__(self, conn):
        """Initialize the service with a :class:`Connection` like object."""
        self._conn = conn

    @classmethod
    def stdio_main(cls):
        """Do the full job of preparing and running an RPC service."""
        conn = Connection.from_fd(*remap_stdio())
        cls(conn).run()

    def run(self):
        """
        Run the service until terminated by either the client or user.

        The service is terminated on user interrupts (Ctrl-C), which might
        or might not be desired.

        """
        import logging
        logging.basicConfig(logLevel=logging.INFO)
        logger = logging.getLogger(__name__)
        try:
            while self._communicate():
                pass
        except KeyboardInterrupt:
            logger.info('User interrupt!')
        finally:
            self._conn.close()

    def _communicate(self):
        """
        Receive and serve one RPC request.

        :returns: ``True`` if the service should continue running.

        """
        try:
            request = self._conn.recv()
        except EOFError:
            return False
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


class LibMadxClient(Client):

    """
    Specialized client for boxing :mod:`cern.cpymad.libmadx` function calls.

    Boxing these MAD-X function calls is necessary due the global nature of
    all state within the MAD-X library.
    """

    def __del__(self):
        """Finalize libmadx if it was started."""
        if self.libmadx.started():
            self.libmadx.finish()

    @property
    class libmadx(object):

        """Wrapper for :mod:`cern.cpymad.libmadx` in a remote process."""

        def __init__(self, client):
            """Store the client connection."""
            self.__client = client

        def __getattr__(self, funcname):
            """Resolve all attribute accesses as remote method calls."""
            def DeferredMethod(*args, **kwargs):
                return self.__client._request('libmadx',
                                              funcname, args, kwargs)
            return DeferredMethod


class LibMadxService(Service):

    """
    Specialized service to dispatch :mod:`cern.cpymad.libmadx` function calls.

    Counterpart for :class:`LibMadxClient`.
    """

    def _dispatch_libmadx(self, funcname, args, kwargs):
        import cern.cpymad.libmadx
        function = getattr(cern.cpymad.libmadx, funcname)
        return function(*args, **kwargs)

if __name__ == '__main__':
    LibMadxService.stdio_main()
