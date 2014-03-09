"""
This is an RPyC server that serves on STDIN/STDOUT.

RPyC is an RPC library that facilitates IPC. This module provides a server
that performs IPC via its STDIN/STDOUT streams. Be sure to start this with
the `python -u` option enabled.

The recommended way to spawn an RPyC server is to use the
:func:`start_server` function.


CAUTION: the service operates in so called 'classic' or 'slave' mode,
meaning that it can be used to perform arbitrary python commands. So be
sure not to leak the STDIN handle to any untrusted source (sockets, etc).
Classic mode is described at:

http://rpyc.readthedocs.org/en/latest/tutorial/tut1.html#tut1


CAUTION: When launching the remote process you should make sure on your own
that the process does not inherit any system resources. On python>=2.7 all
handles will be closed automatically when doing `execv`, i.e. when using
`subprocess.Popen` - but NOT with `multiprocessing.Process`. I don't know
any reliable method that works on python2.6. For more information, see:

http://www.python.org/dev/peps/pep-0446/

"""
__all__ = ['LibMadxClient']

import sys
try:                # python2's cPickle is accelerated version of pickle
    import cPickle as pickle
except ImportError: # python3 automatically import accelerated version
    import pickle


def remap_stdio():
    """
    Setup the streams for an RPC server that communicates on STDIN/STDOUT.

    :returns: old stdin, old stdout
    :rtype: tuple

    The service is run on the STDIN/STDOUT streams that are initially
    present. The file descriptor for STDOUT is then remapped to console
    output, i.e. libraries and modules will still be able to perform
    output.

    """
    import os
    STDIN = 0       # POSIX file descriptor == sys.stdin.fileno()
    STDOUT = 1
    STDERR = 2
    # Duplicate the initial STDIN/STDOUT streams, so they don't get closed
    # when remapping the file descriptors later. These are the streams that
    # are used for IPC with the client:
    ipc_recv = os.fdopen(os.dup(STDIN), 'rb')
    ipc_send = os.fdopen(os.dup(STDOUT), 'wb', 0)
    # virtual file name for console (terminal) IO:
    console = 'con:' if sys.platform == 'win32' else '/dev/tty'
    # Create new python file objects for STDIN/STDOUT and remap the
    # corresponding file descriptors:
    try:
        # Reopen python standard streams. This enables all python modules
        # to use these streams. Note: these objects are currently not
        # flushed automatically when writing to them.
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(console, 'wt', 0)
        # By duplicating the file descriptors to the STDIN/STDOUT file
        # descriptors non-python libraries can make use of these streams as
        # well:
        os.dup2(sys.stdin.fileno(), STDIN)
        os.dup2(sys.stdout.fileno(), STDOUT)
    except (IOError, OSError):
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(os.devnull, 'w', 0)
    return ipc_recv, ipc_send

class Connection(object):
    """
    Pipe-like IPC connection using file objects.
    """
    def __init__(self, recv, send):
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
        self._recv.close()
        self._send.close()
    @property
    def closed(self):
        return self._recv.closed and self._send.closed


# Client side code:
class Client(object):
    def __init__(self, conn):
        self._conn = conn
    def __del__(self):
        self.close()

    @classmethod
    def spawn_subprocess(cls):
        """
        Spawn an RPyC server and establish a connection via its stdio streams.

        :returns: connection to the newly created server
        :rtype: rpyc.Connection

        """
        from subprocess import Popen, PIPE
        args = [sys.executable, '-u', '-m', __name__]
        proc = Popen(args, stdin=PIPE, stdout=PIPE)
        conn = Connection(proc.stdout, proc.stdin)
        return cls(conn)

    def close(self):
        """Close the connection gracefully."""
        try:
            self._conn.send(('close', ()))
        except ValueError:      # already closed
            pass
        self._conn.close()

    def _request(self, kind, *args):
        """Communicate with the remote process."""
        self._conn.send((kind, args))
        return self._dispatch(self._conn.recv())

    def _dispatch(self, response):
        kind, args = response
        handler = getattr(self, '_dispatch_%s' % (kind,))
        return handler(*args)

    def _dispatch_exception(self, exc_info):
        raise exc_info

    def _dispatch_data(self, data):
        return data

class Service(object):
    def __init__(self, conn):
        self._conn = conn

    @classmethod
    def stdio_main(cls):
        """Do the full job of preparing and running an RPC service."""
        cls(Connection(*remap_stdio())).run()

    def run(self):
        import logging
        logging.basicConfig(logLevel=logging.INFO)
        logger = logging.getLogger(__name__)
        try:
            while self._communicate():
                pass
        except KeyboardInterrupt:
            logger.info('User interrupt!')
        finally:
            self.close()

    @property
    def closed(self):
        return self._conn.closed

    def close(self):
        self._conn.close()

    def _communicate(self):
        try:
            request = self._conn.recv()
        except EOFError:
            return False
        else:
            return self._dispatch(request)

    def _dispatch(self, request):
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
                if self.closed:
                    return False
                raise
        return True

    def _dispatch_close(self):
        self.close()

    def _reply_data(self, data):
        self._conn.send(('data', (data,)))

    def _reply_exception(self, exc_info):
        self._conn.send(('exception', (exc_info[1],)))

class LibMadxClient(Client):
    @property
    class libmadx(object):
        """
        Wrapper for cern.madx in a remote process.

        Executes all method calls in the remote process. This is necessary due
        the global nature of the MAD-X library.

        """
        def __init__(self, client):
            self.__client = client
        def __getattr__(self, funcname):
            """Resolve all attribute accesses as remote method calls."""
            def DeferredMethod(*args, **kwargs):
                return self.__client._request('libmadx',
                                              funcname, args, kwargs)
            return DeferredMethod

class LibMadxService(Service):
    def _dispatch_libmadx(self, funcname, args, kwargs):
        import libmadx
        function = getattr(libmadx, funcname)
        return function(*args, **kwargs)

if __name__ == '__main__':
    LibMadxService.stdio_main()
