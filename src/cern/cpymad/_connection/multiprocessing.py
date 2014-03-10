try:
    from _multiprocessing import Connection as _Connection
except ImportError:
    from multiprocessing.connection import Connection as _Connection

class Connection(object):
    @classmethod
    def from_fd(cls, recv, send):
        return _Connection(recv, send)

    @classmethod
    def to_subprocess(cls, args):
        from subprocess import Popen
        from multiprocessing import Pipe
        conn, rcon = Pipe(True)
        proc = Popen(args, stdin=rcon, stdout=rcon)
        rcon.close()
        return conn
