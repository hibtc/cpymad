# The following code makes sure to read all available stdout lines before
# sending more input to MAD-X (ensure the real chronological order!), see:
#       linux:   https://stackoverflow.com/q/375427/650222
#       windows: https://stackoverflow.com/a/34504971/650222
#                https://gist.github.com/techtonik/48c2561f38f729a15b7b

import time
try:                        # Linux
    import fcntl
    from os import O_NONBLOCK
    def set_nonblocking(pipe):
        fd = pipe.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | O_NONBLOCK)

except ImportError:         # Windows
    import msvcrt
    from ctypes import windll, byref, WinError
    from ctypes.wintypes import HANDLE, DWORD, LPDWORD, BOOL

    PIPE_NOWAIT = DWORD(0x00000001)

    # NOTE: SetNamedPipeHandleState works for anonymous pipes as well.
    SetNamedPipeHandleState = windll.kernel32.SetNamedPipeHandleState
    SetNamedPipeHandleState.argtypes = [HANDLE, LPDWORD, LPDWORD, LPDWORD]
    SetNamedPipeHandleState.restype = BOOL

    def set_nonblocking(pipe):
        fd = pipe.fileno()
        hd = msvcrt.get_osfhandle(fd)
        if SetNamedPipeHandleState(hd, byref(PIPE_NOWAIT), None, None) == 0:
            raise OSError(WinError())


class StreamReader:

    """Read stream asynchronously in a worker thread. Note that the worker
    thread will only be active while have entered the `with` context."""

    # NOTE: If MAD-X writes too much output to its STDOUT pipe without any
    # consumer running in parallel, the write will block -- leading to a
    # deadlock. Therefore we *should* ensure to read the pipe while waiting
    # for MAD-X commands to finish. However, any threaded solution I came up
    # with that is able to keep the chronological order had a *huge*
    # performance deficit (every command takes ~100ms) -- which is why we
    # currently stay with a single-threaded solution that keeps the
    # chronological order but could potentially deadlock.

    def __init__(self, stream, callback):
        super().__init__()
        set_nonblocking(stream)
        self.stream = stream
        self.callback = callback

    def __enter__(self):
        pass

    def __exit__(self, *exc_info):
        lines = []
        while True:
            try:
                line = self.stream.readline()
            except IOError:
                break
            if not line:
                break
            lines.append(line.decode('utf-8', 'replace')[:-1])
        if lines:
            self.callback("\n".join(lines))

    def flush(self):
        """Read all data from the remote."""
        with self:
            pass
