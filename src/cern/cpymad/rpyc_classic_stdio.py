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
__all__ = ['start']

import logging, os, sys

import rpyc
import rpyc.utils.classic
import rpyc.utils.factory


def start_server():
    """
    Spawn an RPyC server and establish a connection via its stdio streams.

    :returns: connection to the newly created server
    :rtype: rpyc.Connection

    """
    args = [sys.executable, '-u', '-m', __name__]
    return rpyc.utils.factory.connect_subproc(args, rpyc.SlaveService)

def server_main():
    """
    Create an RPyC slave server that communicates via STDIN and STDOUT.

    NOTE: The service is run on the STDIN/STDOUT streams that are initially
    present. The file descriptor for STDOUT is then remapped to console
    output, i.e. libraries and modules will still be able to perform output.

    """
    STDIN = 0       # POSIX file descriptor == sys.stdin.fileno()
    STDOUT = 1
    STDERR = 2

    # Duplicate the initial STDIN/STDOUT streams, so they don't get closed
    # when remapping the file descriptors later. These are the streams that
    # are used for IPC with the client:
    ipc_recv = os.fdopen(os.dup(STDIN), 'rb')
    ipc_send = os.fdopen(os.dup(STDOUT), 'wb')

    # virtual file name for console (terminal) IO:
    console = 'con:' if sys.platform == 'win32' else '/dev/tty'

    # Create new python file objects for STDIN/STDOUT and remap the
    # corresponding file descriptors:
    try:
        # Reopen python standard streams. This enables all python modules
        # to use these streams. Note: these objects are currently not
        # flushed automatically when writing to them.
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(console, 'w')
        # By duplicating the file descriptors to the STDIN/STDOUT file
        # descriptors non-python libraries can make use of these streams as
        # well:
        os.dup2(sys.stdin.fileno(), STDIN)      
        os.dup2(sys.stdout.fileno(), STDOUT)
    except (IOError, OSError):
        sys.stdin = open(os.devnull, 'r')
        sys.stdout = open(os.devnull, 'w')

    logger = logging.getLogger(__name__)

    # Now create a slave server on this connection and serve all requests:
    conn = rpyc.utils.classic.connect_pipes(ipc_recv, ipc_send)
    try:
        try:
            conn.serve_all()
        except KeyboardInterrupt:
            logger.info('User interrupt!')
    finally:
        conn.close()

if __name__ == '__main__':
    logging.basicConfig(logLevel=logging.INFO)
    server_main()

