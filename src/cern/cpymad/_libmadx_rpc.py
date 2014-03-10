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
__all__ = ['LibMadxClient']

import os
import sys
try:
    # python2's cPickle is an accelerated (C extension) version of pickle:
    import cPickle as pickle
except ImportError:
    # python3's pickle automatically uses the accelerated version and falls
    # back to the python version, see:
    # http://docs.python.org/3.3/whatsnew/3.0.html?highlight=cpickle
    import pickle

def reopen_stdio():
    """
    Reopen the standard input and output streams.

    :obj:`sys.stdin` is mapped to a NULL stream.
    :obj:`sys.stdout` is initialized with the current console.

    CAUTION: the streams are opened in binary mode! This deviates from the
    default on python3!

    """
    STDIN = 0       # POSIX file descriptor == sys.stdin.fileno()
    STDOUT = 1
    # virtual file name for console (terminal) IO:
    console = 'con:' if sys.platform == 'win32' else '/dev/tty'
    # Create new python file objects for STDIN/STDOUT and remap the
    # corresponding file descriptors: Reopen python standard streams. This
    # enables all python modules to use these streams. Note: the stdout
    # buffer length is set to '1', making it line buffered, which behaves
    # like the default in most circumstances.
    sys.stdin = open(os.devnull, 'rt')
    try:
        sys.stdout = open(console, 'wt', 1)
    except (IOError, OSError):
        sys.stdout = open(os.devnull, 'wt', 1)
    # By duplicating the file descriptors to the STDIN/STDOUT file
    # descriptors non-python libraries can make use of these streams as
    # well:
    os.dup2(sys.stdin.fileno(), STDIN)
    os.dup2(sys.stdout.fileno(), STDOUT)


class Connection(object):
    """
    Pipe-like IPC connection using file objects.

    For most purposes this should behave like the connection objects
    returned by :func:`multiprocessing.Pipe`.

    """
    def __init__(self, recv, send):
        """
        Initialize the connection with the given streams.

        :param recv: stream object used for receiving data
        :param send: stream object used for sending data

        """
        self._recv = recv
        self._send = send

    def recv(self):
        """Receive a pickled message from the remote end."""
        return pickle.load(self._recv)

    def send(self, data):
        """Send a pickled message to the remote end."""
        # '-1' instructs pickle to use the latest protocol version. This
        # improves performance by a factor ~50-100 in my tests:
        return pickle.dump(data, self._send, -1)

    def close(self):
        """Close the connection."""
        self._recv.close()
        self._send.close()

    @property
    def closed(self):
        """Check if the connection is fully closed."""
        return self._recv.closed and self._send.closed

    @classmethod
    def from_fd(cls, recv_fd, send_fd):
        """
        Create a :class:`Connection` object from the given file descriptors.

        :param recv: file descriptor used for receiving data
        :param send: file descriptor used for sending data

        """
        return cls(os.fdopen(recv_fd, 'rb', 0),
                   os.fdopen(send_fd, 'wb', 0))

    @classmethod
    def from_stream(cls, recv, send):
        """
        Create a :class:`Connection` object using the given streams.

        :param recv: stream object used for receiving data
        :param send: stream object used for sending data

        The given stream objects invalidated (closed) so they cannot
        accidentally be used anywhere else.

        """
        recv_fd = os.dup(recv.fileno())
        send_fd = os.dup(send.fileno())
        recv.close()
        send.close()
        return cls.from_fd(recv_fd, send_fd)


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
        from subprocess import Popen, PIPE
        args = [sys.executable, '-u', '-m', entry]
        proc = Popen(args, stdin=PIPE, stdout=PIPE)
        conn = Connection.from_stream(proc.stdout, proc.stdin)
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
        conn = Connection.from_stream(sys.stdin, sys.stdout)
        reopen_stdio()
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
        self._conn.send(('exception', (exc_info[1],)))

class LibMadxClient(Client):
    """
    Specialized client for boxing :mod:`cern.cpymad.libmadx` function calls.

    Boxing these MAD-X function calls is necessary due the global nature of
    all state within the MAD-X library.

    """
    @property
    class libmadx(object):
        """Wrapper for :mod:`cern.cpymad.libmadx` in a remote process."""
        def __init__(self, client):
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
