"""
Installation script for cpymad.

Usage:
    python setup.py install --madxdir=/path/to/madx/installation

For more information, see
    http://hibtc.github.io/cpymad/installation
"""

from setuptools import setup, Extension
from distutils.util import get_platform, convert_path
from distutils import sysconfig

import itertools
import sys
from os import path


try:
    # Use Cython if available:
    from Cython.Build import cythonize
except ImportError:
    # Otherwise, use the shipped .c file:
    def cythonize(extensions):
        return extensions


def fix_distutils_sysconfig_mingw():
    """
    When using windows and MinGW, in distutils.sysconfig the compiler (CC) is
    not initialized at all, see http://bugs.python.org/issue2437. The
    following manual fix for this problem may cause other issues, but it's a
    good shot.
    """
    if sysconfig.get_config_var('CC') is None:
        sysconfig._config_vars['CC'] = 'gcc'


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


def remove_arg(args, opt):
    """
    Remove all occurences of ``--PARAM=VALUE`` or ``--PARAM VALUE`` from
    ``args`` and return the corresponding values.
    """
    iterargs = iter(args)
    result = []
    remain = []
    for arg in iterargs:
        if arg == opt:
            result.append(next(iterargs))
            continue
        elif arg.startswith(opt + '='):
            result.append(arg.split('=', 1)[1])
            continue
        remain.append(arg)
    args[:] = remain
    return result


def get_extension_args(argv):
    """Get arguments for C-extension (include pathes, libraries, etc)."""
    # Let's just use the default system headers:
    include_dirs = []
    library_dirs = []
    # Parse command line option: --madxdir=/path/to/madxinstallation. We could
    # use build_ext.user_options instead, but then the --madxdir argument can
    # be passed only to the 'build_ext' command, not to 'build' or 'install',
    # which is a minor nuisance.
    for dir_ in remove_arg(argv, '--madxdir'):
        prefix = path.expanduser(dir_)
        include_dirs += [path.join(prefix, 'include')]
        library_dirs += [path.join(prefix, 'lib'),
                         path.join(prefix, 'lib64')]

    # Windows:  win32/win-amd64
    # Linux:    linux-x86_64/...
    # Mac:      darwin*
    platform = get_platform()

    static = '--static' in argv
    if static:
        argv.remove('--static')
        libraries = ['madx', 'ptc', 'gc-lib',
                     'stdc++', 'gfortran', 'quadmath']
        # NOTE: If MAD-X was built with BLAS/LAPACK, you must manually provide
        # arguments `python setup.py build_ext -lblas -llapack`!
    else:
        libraries = ['madx']

    if static and platform.startswith('linux'):
        libraries += ['X11']

    return dict(
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=['-std=gnu99'],
    )


def get_setup_args(argv):
    """Accumulate metadata for setup."""
    extension_args = get_extension_args(argv)
    long_description = get_long_description()
    metadata = exec_file('cpymad/__init__.py')
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
        ext_modules = cythonize([
            Extension('cpymad.libmadx',
                      sources=["cpymad/libmadx.pyx"],
                      **extension_args),
        ]),
        packages = [
            "cpymad",
        ],
        include_package_data=True, # include files matched by MANIFEST.in
        install_requires=[
            'setuptools>=18.0',
            'numpy',
            'PyYAML',
            'minrpc>=0.0.7',
        ],
    )


def main():
    """
    Execute setup.

    Note that this operation is controlled via sys.argv and has side-effects
    both on sys.argv as well as some parts of the distutils package.
    """
    fix_distutils_sysconfig_mingw()
    setup_args = get_setup_args(sys.argv)
    setup(**setup_args)


if __name__ == '__main__':
    main()
