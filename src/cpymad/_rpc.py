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

from minrpc.client import Client, RemoteProcessCrashed, RemoteProcessClosed


__all__ = [
    'LibMadxClient',
    'RemoteProcessCrashed',
    'RemoteProcessClosed',
]


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
        return self.get_module('cpymad.libmadx')
