"""
Simple RPC module for libmadx.

This module is needed to execute several instances of the `libmadx` module
in remote processes and communicate with them via remote procedure calls
(RPC). Use the :meth:`LibMadxClient.spawn` to create a new instance.

The remote backend is needed due to the fact that cpymad.libmadx is a low
level binding to the MAD-X library which in turn uses global variables.
This means that the cpymad.libmadx module has to be loaded within remote
processes in order to deal with several isolated instances of MAD-X in
parallel.

Furthermore, this can be used as a security enhancement: if dealing with
unverified input, we can't be sure that a faulty MAD-X function
implementation will give access to a secure resource. This can be executing
all library calls within a subprocess that does not inherit any handles.

More importantly: the MAD-X program crashes on the tinyest error. Boxing it
in a subprocess will prevent the main process from crashing as well.

CAUTION:

The service communicates with the remote end via pickling, i.e. both ends
can execute arbitrary code on the other side. This means that the remote
process can not be used to safely execute unsafe python code.
"""

from __future__ import absolute_import

__all__ = [
    'LibMadxClient',
    'RemoteProcessCrashed',
]

import traceback
import os
import subprocess
import sys

try:
    # python2's cPickle is an accelerated (C extension) version of pickle:
    import cPickle as pickle
except ImportError:
    # python3's pickle automatically uses the accelerated version and falls
    # back to the python version, see:
    # http://docs.python.org/3.3/whatsnew/3.0.html?highlight=cpickle
    import pickle


_win = sys.platform == 'win32'


def _nop(x):
    """NO-OP: do nothing, just return x."""
    return x


class RemoteProcessClosed(RuntimeError):
    """The MAD-X remote process has already been closed."""
    pass


class RemoteProcessCrashed(RuntimeError):
    """The MAD-X remote process has crashed."""
    pass


if _win:
    import msvcrt


    try:                    # python2
        import _subprocess as _winapi
        # _subprocess.DuplicateHandle and _subprocess.CreatePipe return a
        # handle type with .Close() and .Detach() methods:
        Handle = _nop
    except ImportError:     # python3
        import _winapi
        # _winapi.DuplicateHandle and _winapi.CreatePipe return plain
        # integers which need to be wrapped in subprocess.Handle to make
        # them closable:
        Handle = subprocess.Handle


    def _make_inheritable(handle):
        """Return inheritable Handle, close the original."""
        # new handles are created uninheritable by default, but they can be
        # made inheritable on duplication:
        current_process = _winapi.GetCurrentProcess()
        dup = Handle(_winapi.DuplicateHandle(
            current_process,
            handle,
            current_process,
            0,
            True,
            _winapi.DUPLICATE_SAME_ACCESS))
        handle.Close()
        return dup


    def _pipe():
        """Create a unidirectional pipe."""
        # use _winapi.CreatePipe on windows, just like subprocess.Popen
        # does when requesting PIPE streams. This is the easiest and most
        # reliable method I have tested so far:
        recv, send = _winapi.CreatePipe(None, 0)
        return Handle(recv), Handle(send)


    def _open(handle):
        """
        Open a file descriptor for the specified HANDLE (int -> int).

        Closing the file descriptor will also close the handle.
        """
        return msvcrt.open_osfhandle(handle, 0)


    def _close(handle):
        """Close the :class:`Handle` object."""
        handle.Close()


    def _detach(handle):
        """Return HANDLE after detaching it from a :class:`Handle` object."""
        return handle.Detach()


# POSIX
else:
    try:
        from os import set_inheritable
    except ImportError:         # python2
        # on POSIX/python2 file descriptors are inheritable by default:
        _make_inheritable = _nop
    else:                       # python3
        def _make_inheritable(fd):
            """Return inheritable file descriptor, close the original."""
            dup = os.dup(fd)
            os.close(fd)
            set_inheritable(dup, True)
            return dup


    _pipe = os.pipe


    # handles are just file descriptors on POSIX:
    _open = _nop
    _close = os.close
    _detach = _nop


def _close_all_but(keep):
    """Close all but the given file descriptors."""
    # first, let the garbage collector run, it may find some unreachable
    # file objects (on posix forked processes) and close them:
    import gc
    gc.collect()
    # close all ranges in between the file descriptors to be kept:
    keep = sorted(set([-1] + keep + [subprocess.MAXFD]))
    for s, e in zip(keep[:-1], keep[1:]):
        if s+1 < e:
            os.closerange(s+1, e)


class Connection(object):

    """
    Pipe-like IPC connection using file objects.

    For most purposes this should behave like the connection objects
    returned by :func:`multiprocessing.Pipe`.

    This class combines two orthogonal functionalities. In general this is
    bad practice, meaning the class should be refactored into two classes,
    but for our specific purpose this will do it.

    - build a bidirectional stream from two unidirectional streams
    - build a serialized connection from pure data streams (pickle)
    """

    def __init__(self, recv, send):
        """Create duplex connection from two unidirectional streams."""
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
        """Create a connection from two file descriptors."""
        return cls(os.fdopen(recv_fd, 'rb', 0),
                   os.fdopen(send_fd, 'wb', 0))


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
        # create two unidirectional pipes for communication with the
        # subprocess.  the remote end needs to be inheritable, the local
        # end is not inheritable by default (or will be closed by
        # _close_all_but on POSIX/py2):
        local_recv, _remote_send = _pipe()
        _remote_recv, local_send = _pipe()
        remote_recv = _make_inheritable(_remote_recv)
        remote_send = _make_inheritable(_remote_send)
        args = [sys.executable, '-u', '-m', __name__]
        args += [str(int(remote_recv)),
                 str(int(remote_send))]
        proc = subprocess.Popen(args, close_fds=False, **Popen_args)
        # close handles that are not used in this process:
        _close(remote_recv)
        _close(remote_send)
        conn = Connection.from_fd(_open(_detach(local_recv)),
                                  _open(_detach(local_send)))
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
        passed_handles = [int(arg) for arg in args]
        _close_all_but([sys.stdin.fileno(),
                        sys.stdout.fileno(),
                        sys.stderr.fileno()] + passed_handles)
        hrecv, hsend = passed_handles[:2]
        conn = Connection.from_fd(_open(hrecv),
                                  _open(hsend))
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
    Specialized client for boxing :mod:`cpymad.libmadx` function calls.

    Boxing these MAD-X function calls is necessary due the global nature of
    all state within the MAD-X library.
    """

    def close(self):
        """Finalize libmadx if it is running."""
        try:
            if self.libmadx.is_started():
                self.libmadx.finish()
        except (RemoteProcessClosed, RemoteProcessCrashed):
            pass
        super(LibMadxClient, self).close()

    @property
    def libmadx(self):
        return self.modules['cpymad.libmadx']

    @property
    class modules(object):

        """Provides access to all modules in the remote process."""

        def __init__(self, client):
            self.__client = client

        def __getitem__(self, key):
            """Get a RemoteModule object by module name."""
            return RemoteModule(self.__client, key)


class RemoteModule(object):

    """Wrapper for :mod:`cpymad.libmadx` in a remote process."""

    def __init__(self, client, module):
        """Store the client connection."""
        self.__client = client
        self.__module = module

    def __getattr__(self, funcname):
        """Resolve all attribute accesses as remote function calls."""
        def DeferredMethod(*args, **kwargs):
            return self.__client._request('function_call', self.__module,
                                          funcname, args, kwargs)
        return DeferredMethod


class LibMadxService(Service):

    """
    Specialized service to dispatch :mod:`cpymad.libmadx` function calls.

    Counterpart for :class:`LibMadxClient`.
    """

    def _dispatch_function_call(self, modname, funcname, args, kwargs):
        """Execute any static function call in the remote process."""
        # As soon as we drop support for python2.6, we should replace this
        # with importlib.import_module:
        module = __import__(modname, None, None, '*')
        function = getattr(module, funcname)
        return function(*args, **kwargs)


if __name__ == '__main__':
    LibMadxService.stdio_main(sys.argv[1:])
