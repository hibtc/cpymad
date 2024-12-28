from __future__ import unicode_literals
import sys
import warnings


__title__ = 'cpymad'
__version__ = '1.17.0'

__summary__ = 'Cython binding to MAD-X'
__uri__ = 'https://github.com/hibtc/cpymad'

__credits__ = """
Current cpymad maintainer:

    - Thomas Gläßle <t_glaessle@gmx.de>

Initial pymad creators:

    - Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>
    - Kajetan Fuchsberger <Kajetan.Fuchsberger@cern.ch>
"""


if sys.version_info < (3, 9):
    _unsupported_version = (
        "Support for python 3.8 and below will be removed in a future release!\n"
        "If you need continued support for an older version, let us know at:\n"
        "  https://github.com/hibtc/cpymad/issues")
    warnings.warn(_unsupported_version, DeprecationWarning)


def _read_text(package, resource, **kwargs):
    try:
        from importlib.resources import files
        return files(package).joinpath(resource).read_text(**kwargs)
    except ImportError:
        from importlib.resources import read_text
        return read_text(package, resource, **kwargs)


def get_copyright_notice() -> str:
    return _read_text('cpymad.COPYING', 'cpymad.rst', encoding='utf-8')
