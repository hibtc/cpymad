from multiprocessing.dummy import Pool as ThreadPool


# The following code makes sure to read all available stdout lines before
# sending more input to MAD-X (ensure the real chronological order!), see:
#       linux:   https://stackoverflow.com/q/375427/650222
#       windows: https://stackoverflow.com/a/34504971/650222
#                https://gist.github.com/techtonik/48c2561f38f729a15b7b

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


class AsyncReader:

    """Read stream asynchronously in a worker thread. Note that the worker
    thread will only be active while have entered the `with` context."""

    def __init__(self, stream, callback):
        set_nonblocking(stream)
        self.pool = ThreadPool(1)
        self.stream = stream
        self.callback = callback

    def __enter__(self):
        self.stop = False
        self.result = self.pool.apply_async(self._read_thread)

    def __exit__(self, *exc_info):
        self.stop = True
        output_lines = self.result.get()
        if output_lines:
            self.callback(b''.join(output_lines))

    def _read_thread(self):
        lines = []
        stop = False
        while True:
            try:
                line = self.stream.readline()
            except IOError:
                if stop:
                    return lines
                # do one more iteration, this prevents missing output due to
                # unfortunate thread scheduling:
                if self.stop:
                    stop = True
                continue
            if not line:
                return lines
            lines.append(line)


class TextCallback:

    """Decode bytes and pass to callback."""

    def __init__(self, callback, encoding='utf-8', errors='replace'):
        self.callback = callback
        self.encoding = encoding
        self.errors = errors

    def __call__(self, data):
        text = data.decode(self.encoding, errors=self.errors)
        return self.callback(text)
