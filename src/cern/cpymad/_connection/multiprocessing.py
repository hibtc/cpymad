try:                    # python2
    from _multiprocessing import Connection as _Connection
except ImportError:     # python3
    from multiprocessing.connection import Connection as _Connection

class Connection(object):
    """
    Pipe-like IPC connection using file objects.
    
    This uses the same connection class as the objects returned by
    :func:`multiprocessing.Pipe`.

    NOTE: On unix this works with arbitrary file descriptors, on windows only
    sockets can be used.

    """
    @classmethod
    def from_fd(cls, recv, send):
        """
        Create a connection from the given file descriptors.

        :param recv: file descriptor used for receiving data
        :param send: file descriptor used for sending data

        NOTE: On unix this works with arbitrary file descriptors, on windows
        only sockets can be used.

        """
        return _Connection(recv, send)

    @classmethod
    def to_subprocess(cls, args):
        """
        Establish connection to a new remote process.

        :param list args: arguments for the subprocess
        :returns: connection to subprocess

        CAUTION: This doesn't work on windows!

        """
        from subprocess import Popen
        from multiprocessing import Pipe
        conn, rcon = Pipe(True)
        proc = Popen(args, stdin=rcon, stdout=rcon)
        rcon.close()
        return conn
