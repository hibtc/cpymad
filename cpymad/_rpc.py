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

import sys

from cpymad.rpc_util import service
from cpymad.rpc_util import client
from cpymad.rpc_util.client import RemoteProcessCrashed, RemoteProcessClosed


__all__ = [
    'LibMadxClient',
    'RemoteProcessCrashed',
    'RemoteProcessClosed',
]


class LibMadxClient(client.Client):

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


class LibMadxService(service.Service):

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
