import os
try:
    # python2's cPickle is an accelerated (C extension) version of pickle:
    import cPickle as pickle
except ImportError:
    # python3's pickle automatically uses the accelerated version and falls
    # back to the python version, see:
    # http://docs.python.org/3.3/whatsnew/3.0.html?highlight=cpickle
    import pickle

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

    @classmethod
    def to_subprocess(cls, args):
        from subprocess import Popen, PIPE
        proc = Popen(args, stdin=PIPE, stdout=PIPE)
        conn = cls.from_stream(proc.stdout, proc.stdin)
        return conn
