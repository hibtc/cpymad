# encoding: utf-8
from __future__ import unicode_literals

__title__ = 'cpymad'
__version__ = '0.15.1'

__summary__ = 'Cython binding to MAD-X'
__uri__ = 'https://github.com/hibtc/cpymad'

__author__ = 'PyMAD developers'
__author_email__ = 'pymad@cern.ch'
__maintainer__ = 'Thomas Gläßle'
__maintainer_email__ = 't_glaessle@gmx.de'

__support__ = __maintainer_email__

__license__ = 'GPLv3+'
__copyright__ = 'See cpymad/COPYING folder or cpymad.get_copyright_notice()'

__credits__ = """
Current cpymad maintainer:

    - Thomas Gläßle <t_glaessle@gmx.de>

Initial pymad creators:

    - Yngve Inntjore Levinsen <Yngve.Inntjore.Levinsen@cern.ch>
    - Kajetan Fuchsberger <Kajetan.Fuchsberger@cern.ch>
"""

__classifiers__ = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'License :: OSI Approved :: Apache Software License',
    'License :: Other/Proprietary License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering :: Physics',
]


def get_copyright_notice():
    from pkg_resources import resource_string
    return resource_string('cpymad', 'COPYING/cpymad.rst')
