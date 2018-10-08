# encoding: utf-8
"""
Installation script for cpymad.

Usage:
    python setup.py install --madxdir=/path/to/madx/installation

For more information, see
    http://hibtc.github.io/cpymad/installation
"""

from __future__ import print_function

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.util import get_platform, convert_path
from distutils import sysconfig
from distutils.errors import PreprocessError, CompileError

from textwrap import dedent
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


def fix_distutils_sysconfig_mingw():
    """
    When using windows and MinGW, in distutils.sysconfig the compiler (CC) is
    not initialized at all, see http://bugs.python.org/issue2437. The
    following manual fix for this problem may cause other issues, but it's a
    good shot.
    """
    if sysconfig.get_config_var('CC') is None:
        sysconfig._config_vars['CC'] = 'gcc'


class build_ext(_build_ext):

    # We parse command line options using our own mechanim… We could use
    # build_ext.user_options instead, but then these parameters can be passed
    # only to the 'build_ext' command, not to 'build', 'develop', or
    # 'install'…
    options = {
        'madxdir':  'arg',
        'static':   'opt',
        'shared':   'opt',
        'lapack':   'opt',
        'blas':     'opt',
        'X11':      'opt',
    }
    optvals = {}

    def build_extension(self, ext):
        ext.__dict__.update(get_extension_args(**self.get_optvals()))
        try:
            # Return if the extension has already been built or MAD-X is
            # available. This prevents us from unnecessary work for example
            # when performing the `install` stage without passing `--madxdir`
            # after already having done `build_ext` successfully:
            return _build_ext.build_extension(self, ext)
        # NOTE that we don't catch LinkerError since that indicates that MAD-X
        # is available but somehow missconfigured and we should never attempt
        # to download/rebuild in that case:
        except (PreprocessError, CompileError):
            # Stop if the user had specified `--madxdir` or if the error is
            # not due to the inavailability of MAD-X. In these cases we should
            # not start trying to build MAD-X:
            if self.has_madx():
                raise
        print(dedent("""

        Could not find MAD-X development files.

        We will now try to download and build MAD-X for you. This may
        take several minutes (depending on your internet connection
        and overall system performance), and requires recent versions
        of the following programs:

            - cmake
            - gcc
            - gfortran

        In case of problems, please build MAD-X manually and then pass
        the installation directory using the `--madxdir` argument to
        `setup.py`. For further information please check the
        installation instructions at:

            http://hibtc.github.io/cpymad/installation/index.html

        """), file=sys.stderr)
        # If unspecified, let's build MAD-X without X11 (as this removes
        # potential linking complications):
        if self.optvals['X11'] is None:
            self.optvals['X11'] = False
        self.build_madx()
        ext.__dict__.update(get_extension_args(**self.get_optvals()))
        return _build_ext.build_extension(self, ext)

    def has_madx(self):
        return self.optvals['madxdir'] or self.check_dependency(dedent("""
        #include "madX/madx.h"
        int main(int argc, char* argv[])
        {
            madx_start();
            return 0;
        }
        """))

    def build_madx(self):
        from utils.build_madx import install_madx
        optvals = self.get_optvals()
        self.optvals['madxdir'] = install_madx(
            prefix=self.build_temp,
            static=optvals['static'],
            shared=optvals['shared'],
            X11=optvals['X11'])

    def check_dependency(self, c_code):
        """Check if an external library can be found by trying to compile a
        small program."""
        # This method is needed to prevent us from downloading and building
        # MAD-X on every error in the C code itself during development even
        # if MAD-X is available.
        print(dedent("""

        No `--madxdir` specified. We will now try to check if the MAD-X
        development files are already installed in a system directory.

        If you have installed MAD-X in a non-standard or user directory,
        please pass the `--madxdir=PATH` argument to `setup.py`.

        """), file=sys.stderr)
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp(prefix='tmp_cpymad_madx_')
        tmp_bin = os.path.join(tmp_dir, 'test_madx')
        tmp_src = tmp_bin + '.c'
        with open(tmp_src, 'w') as f:
            f.write(c_code)
        ext_args = get_extension_args(**self.get_optvals())
        try:
            self.compiler.compile(
                [tmp_src],
                output_dir=tmp_dir,
                include_dirs=ext_args['include_dirs'],
                extra_postargs=ext_args['extra_compile_args'])
            return True
        except (PreprocessError, CompileError):
            return False
        finally:
            shutil.rmtree(tmp_dir)

    def get_optvals(self):
        optvals = self.optvals.copy()
        if optvals['static'] is None:
            optvals['static'] = IS_WIN
        # If unspecified, assume MAD-X was built without disabling X11:
        if optvals['X11'] is None:
            optvals['X11'] = not IS_WIN
        return optvals


def read_file(path):
    """Read a file in binary mode."""
    with open(convert_path(path), 'rb') as f:
        return f.read()


def exec_file(path):
    """Execute a python file and return the `globals` dictionary."""
    namespace = {}
    exec(read_file(path), namespace, namespace)
    return namespace


def first_sections(document, heading_type, num_sections):
    cur_section = 0
    prev_line = None
    for line in document.splitlines():
        if line.startswith(heading_type*3) and set(line) == {heading_type}:
            cur_section += 1
            if cur_section == num_sections:
                break
        if prev_line is not None:
            yield prev_line
        prev_line = line


def get_long_description():
    """Compose a long description for PyPI."""
    long_description = None
    try:
        long_description = read_file('README.rst').decode('utf-8')
        changelog = read_file('CHANGES.rst').decode('utf-8')
        changelog = "\n".join(first_sections(changelog, '=', 4)) + """
Older versions
==============

The full changelog is available online in CHANGES.rst_.

.. _CHANGES.rst: https://github.com/hibtc/cpymad/blob/master/CHANGES.rst
"""
        long_description += '\n' + changelog
    except (IOError, UnicodeDecodeError):
        pass
    return long_description


def get_extension_args(madxdir, shared, static, **libs):
    """Get arguments for C-extension (include pathes, libraries, etc)."""
    include_dirs = []
    library_dirs = []
    if madxdir is None:
        searchpaths = [
            os.path.expanduser('~/.local'),
            '/opt/madx',
            '/usr/local',
        ]
        for d in searchpaths:
            if os.path.exists(os.path.join(
                    d, 'include', 'madX', 'madx.h')):
                madxdir = d
                break

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


def get_setup_args(optvals):
    """Accumulate metadata for setup."""
    long_description = get_long_description()
    metadata = exec_file('src/cpymad/__init__.py')
    return dict(
        name='cpymad',
        version=metadata['__version__'],
        description=metadata['__summary__'],
        long_description=long_description,
        author=metadata['__author__'],
        author_email=metadata['__author_email__'],
        maintainer=metadata['__maintainer__'],
        maintainer_email=metadata['__maintainer_email__'],
        url=metadata['__uri__'],
        license=metadata['__license__'],
        classifiers=metadata['__classifiers__'],
        ext_modules=cythonize([
            Extension('cpymad.libmadx',
                      sources=["src/cpymad/libmadx.pyx"],
                      libraries=['madx']),
        ]),
        packages=find_packages('src'),
        package_dir={'': 'src'},
        include_package_data=True,  # include files matched by MANIFEST.in
        install_requires=[
            'setuptools>=18.0',
            'numpy',
            'minrpc>=0.0.8',
        ],
        cmdclass={'build_ext': type('build_ext', (build_ext, object), {
            'optvals': optvals,
        })},
    )


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
    from utils.clopts import parse_opts
    fix_distutils_sysconfig_mingw()
    optvals = parse_opts(sys.argv, build_ext.options)
    setup(**get_setup_args(optvals))
