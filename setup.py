# encoding: utf-8
"""
Setup script for cpymad.

Usage:
    python setup.py bdist_wheel --madxdir=/path/to/madx/installation

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

from setuptools import setup, find_packages, Extension
from distutils.util import get_platform
from distutils import sysconfig

import sys
import os

try:
    # Use Cython if available:
    from Cython.Build import cythonize
except ImportError:
    # Otherwise, use the shipped .c file:
    def cythonize(extensions):
        return extensions

# Windows:  win32/win-amd64
# Linux:    linux-x86_64/...
# Mac:      darwin*
IS_WIN = get_platform().startswith('win')


# We parse command line options using our own mechanim. We could use
# build_ext.user_options instead, but then these parameters can be passed
# only to the 'build_ext' command, not to 'build', 'develop', or
# 'install'.
OPTIONS = {
    'madxdir':  'arg',
    'static':   'opt',
    'shared':   'opt',
    'lapack':   'opt',
    'blas':     'opt',
    'X11':      'opt',
}


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
    if libs.get('X11') is None:
        libs['X11'] = not IS_WIN
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
        libraries += ['ptc', 'gc-lib', 'stdc++', 'gfortran', 'quadmath']
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
    from utils.clopts import parse_opts
    fix_distutils_sysconfig_mingw()
    optvals = parse_opts(sys.argv, OPTIONS)
    metadata = exec_file('src/cpymad/__init__.py')
    setup(
        name='cpymad',
        version=metadata['__version__'],
        description=metadata['__summary__'],
        ext_modules=cythonize([
            Extension('cpymad.libmadx',
                      sources=["src/cpymad/libmadx.pyx"],
                      **get_extension_args(**optvals)),
        ]),
        packages=find_packages('src'),
        package_dir={'': 'src'},
        zip_safe=False,             # zip is bad for redistributing shared libs
        include_package_data=True,  # include files matched by MANIFEST.in
        install_requires=[
            'importlib_resources',
            'numpy',
            'minrpc>=0.0.8',
        ],
    )
