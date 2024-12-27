# encoding: utf-8
"""
Setup script for cpymad.

This script is meant only for packagers and developers and can be used to
install cpymad or create installers assuming you have already built MAD-X
beforehand.

For more information, see
    http://hibtc.github.io/cpymad/installation
"""

# Now that we distribute wheels for most supported platforms we do not attempt
# to build MAD-X automatically! These are the preferred method of installation
# for most users while for packagers/developers it is preferrable to have
# direct control over the MAD-X build process itself. Therefore, there are few
# who would actually profit from automatic setup. Furthermore, building MAD-X
# in setup.py adds a lot of complexity. If you don't believe me, take a look
# at the commit that first added this paragraph (can be identified using `git
# blame`) and the simplifications that were possible in the following commits.

from setuptools import setup, Extension
from distutils.util import get_platform
from distutils import sysconfig
from argparse import ArgumentParser

import sys
import os

try:
    # Use Cython if available:
    from Cython.Build import cythonize
except ImportError:
    # Otherwise, use the shipped .c file:
    def cythonize(extensions):
        return extensions

# Windows:              win32/win-amd64
# Linux:                linux-x86_64/...
# Mac Intel:            darwin*
# Mac Apple Silicon:    *-arm64
platform = get_platform()
IS_WIN = platform.startswith('win')
IS_ARM = platform.startswith('linux-aarch') or platform.endswith('arm64')


# We parse command line options using our own mechanim. We could use
# build_ext.user_options instead, but then these parameters can be passed
# only to the 'build_ext' command, not to 'build', 'develop', or
# 'install'.
def command_line_options():
    usage = 'setup.py <command> [options]'
    parser = ArgumentParser(description=__doc__, usage=usage)
    parser.add_argument(
        '--madxdir', dest='madxdir',
        default=os.environ.get('MADXDIR'),
        help='MAD-X installation prefix')
    option(parser, 'static', 'do {NOT}use static linkage')
    option(parser, 'shared', 'MAD-X was {NOT}built with BUILD_SHARED_LIBS')
    option(parser, 'lapack', 'MAD-X was {NOT}built with LAPACK')
    option(parser, 'blas', 'MAD-X was {NOT}built with BLAS')
    option(parser, 'X11', 'MAD-X was {NOT}built with MADX_X11')
    option(parser, 'quadmath', 'do {NOT}link against libquadmath')
    return parser


def option(parser, name, descr):
    """Add a negatable option to parser."""
    env_var = os.environ.get(name.upper())
    default = bool(int(env_var)) if env_var else None
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, dest=name,
        default=default,
        action='store_true', help=descr.format(NOT=''))
    group.add_argument(
        '--no-' + name, dest=name,
        action='store_false', help=descr.format(NOT='not '))
    return group


def fix_distutils_sysconfig_mingw():
    """
    When using windows and MinGW, in distutils.sysconfig the compiler (CC) is
    not initialized at all, see http://bugs.python.org/issue2437. The
    following manual fix for this problem may cause other issues, but it's a
    good shot.
    """
    if sysconfig.get_config_var('CC') is None:
        sysconfig._config_vars['CC'] = 'gcc'


def exec_file(path):
    """Execute a python file and return the `globals` dictionary."""
    namespace = {}
    with open(path, 'rb') as f:
        exec(f.read(), namespace, namespace)
    return namespace


def get_extension_args(madxdir, shared, static, **libs):
    """Get arguments for C-extension (include pathes, libraries, etc)."""
    # libquadmath isn't available on aarch64, see #101:
    if libs.get('quadmath') is None:
        libs['quadmath'] = not IS_ARM
    include_dirs = []
    library_dirs = []

    if madxdir:
        prefix = os.path.expanduser(madxdir)
        include_dirs += [os.path.join(prefix, 'include')]
        library_dirs += [os.path.join(prefix, 'lib'),
                         os.path.join(prefix, 'lib64')]

    libraries = ['madx']
    if not shared:
        # NOTE: If MAD-X was built with BLAS/LAPACK, you must manually provide
        # arguments `python setup.py build_ext -lblas -llapack`!
        libraries += ['DISTlib', 'ptc', 'gc-lib', 'stdc++', 'gfortran']
        libraries += [lib for lib, use in libs.items() if use]

    link_args = ['--static'] if static and not IS_WIN else []

    return dict(
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs if shared else [],
        extra_compile_args=['-std=gnu99'],
        extra_link_args=link_args,
    )


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    fix_distutils_sysconfig_mingw()
    options, sys.argv[1:] = command_line_options().parse_known_args()
    # NOTE: The "metadata" parameters for setup() may appear to be redundant
    # but are curently required on setuptools<61.0 and hence for python3.6
    # for which setuptools>=60 is not available. See branch `drop-py36` for
    # removal.
    metadata = exec_file('src/cpymad/__init__.py')
    setup(
        name='cpymad',
        version=metadata['__version__'],
        description=metadata['__summary__'],
        ext_modules=cythonize([
            Extension('cpymad.libmadx',
                      sources=["src/cpymad/libmadx.pyx"],
                      **get_extension_args(**options.__dict__)),
        ]),
        packages=['cpymad', 'cpymad.COPYING'],
        package_dir={'': 'src'},
        zip_safe=False,             # zip is bad for redistributing shared libs
        include_package_data=True,  # include files matched by MANIFEST.in
        install_requires=[
            'numpy',
            'minrpc>=0.0.8',
        ],
    )
