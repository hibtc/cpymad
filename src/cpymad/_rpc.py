"""
Simple RPC module for libmadx.

This module is used to execute several instances of the `libmadx` module
in remote processes and communicate with them via remote procedure calls
(RPC). Use the :meth:`LibMadxClient.spawn` to create a new instance.

The remote backend is needed due to the fact that cpymad.libmadx is a low
level binding to the MAD-X library which in turn uses global variables.
This means that the cpymad.libmadx module has to be loaded within remote
processes in order to deal with several isolated instances of MAD-X in
parallel. Furthermore, running MAD-X in a separate process shields the main
python process from crashes due to incorrect use of MAD-X.
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
