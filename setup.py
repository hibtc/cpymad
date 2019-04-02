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
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.util import get_platform
from distutils import sysconfig

import os

# Windows:  win32/win-amd64
# Linux:    linux-x86_64/...
# Mac:      darwin*
IS_WIN = get_platform().startswith('win')


class build_ext_cythonize(_build_ext):

    # NOTE: If MAD-X was built with X11/BLAS/LAPACK, you must manually
    # provide arguments `python setup.py build_ext -lX11 -lblas -llapack`!

    user_options = _build_ext.user_options + [
        ('madxdir=', 'M', 'Specify the installation prefix used for MAD-X'),
        ('shared', None, 'MAD-X was built as shared library'),
        ('static', None, 'Link statically (not recommended)'),
    ]

    def initialize_options(self):
        self.madxdir = None
        self.shared = False
        self.static = False
        _build_ext.initialize_options(self)

    def finalize_options(self):
        print(self.madxdir)
        exit()
        # Everyone who wants to build cpymad from source needs cython because
        # the generated C code is partially incompatible across different
        # python versions, see: 77c5012e "Recythonize for each python
        # version":
        from Cython.Build import cythonize
        self.extensions = cythonize([
            Extension(ext.name, ext.sources, **get_extension_args(
                self.madxdir, self.shared, self.static))
            for ext in self.extensions
        ])
        # When using windows and MinGW, in distutils.sysconfig the C compiler
        # (CC) is not initialized at all, see http://bugs.python.org/issue2437.
        if sysconfig.get_config_var('CC') is None:
            sysconfig._config_vars['CC'] = 'gcc'
        _build_ext.finalize_options(self)


def get_extension_args(madxdir, shared, static):
    """Get arguments for C-extension (include pathes, libraries, etc)."""
    include_dirs = []
    library_dirs = []

    if madxdir:
        prefix = os.path.expanduser(madxdir)
        include_dirs += [os.path.join(prefix, 'include')]
        library_dirs += [os.path.join(prefix, 'lib')]

    libraries = ['madx'] + ([] if shared else [
        'ptc', 'gc-lib', 'stdc++', 'gfortran', 'quadmath'])

    link_args = ['--static'] if static and not IS_WIN else []

    return dict(
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs if shared else [],
        extra_compile_args=['-std=gnu99'],
        extra_link_args=link_args,
    )


setup(
    ext_modules=[
        Extension('cpymad.libmadx', ["src/cpymad/libmadx.pyx"]),
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    cmdclass={'build_ext': build_ext_cythonize},
)
