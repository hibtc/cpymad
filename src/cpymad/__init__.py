# encoding: utf-8
from __future__ import unicode_literals

__title__ = 'cpymad'
__version__ = '1.4.0'

__summary__ = 'Cython binding to MAD-X'
__uri__ = 'https://github.com/hibtc/cpymad'

__credits__ = """
Current cpymad maintainer:

    - Thomas Gläßle <t_glaessle@gmx.de>

Initial pymad creators:

    - Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>
    - Kajetan Fuchsberger <Kajetan.Fuchsberger@cern.ch>
"""


def get_copyright_notice():
    from importlib_resources import read_text
    return read_text('cpymad.COPYING', 'cpymad.rst', encoding='utf-8')
