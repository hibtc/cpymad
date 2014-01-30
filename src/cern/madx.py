"""
Python wrapper for the MAD-X library.

This is only a backward compatibility alias for the real module which is
located at cern.cpymad.madx. 

"""
__all__ = ['madx']

from cern.cpymad.madx import madx
