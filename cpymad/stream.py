from threading import Thread, Event
from queue import Queue, Empty


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


class AsyncReader:

    """Read stream asynchronously in a worker thread."""

    def __init__(self, stream, callback):
        super().__init__()
        self.queue = Queue()
        self.event = Event()
        self.stream = stream
        self.callback = callback
        self.thread = Thread(target=self._read_thread)
        self.thread.daemon = True   # don't block program exit
        self.thread.start()

    def read_all(self):
        """Read all data currently available in the queue."""
        while True:
            try:
                x = self.queue.get_nowait()
            except Empty:
                return
            yield x

    def _read_thread(self):
        set_nonblocking(self.stream)
        while True:
            try:
                line = self.stream.readline()
            except IOError:
                self.event.set()
                time.sleep(0.001)
                continue
            if not line:
                return
            self.queue.put(line.decode('utf-8', 'replace')[:-1])
            self.callback(self)

    def flush(self):
        """Read all data from the remote."""
        self.event.clear()
        self.event.wait()
        if not self.queue.empty():
            self.callback(self)         # emit signal in direct mode!
