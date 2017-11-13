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
        lib_path_candidates = [path.join(prefix, 'lib'),
                               path.join(prefix, 'lib64')]
        include_dirs += [path.join(prefix, 'include')]
        library_dirs += list(filter(path.isdir, lib_path_candidates))
    # Determine shared libraries:
    # NOTE: using ``distutils.util.get_platform()`` rather than
    # ``sys.platform`` or ``platform.system()`` or even ``os.name`` and
    # ``os.uname()`` to handle cross-builds:
    platform = get_platform()
    # win32/win-amd64:
    if platform.startswith('win'):
        libraries = ['madx', 'ptc', 'gc-lib',
                     'stdc++', 'gfortran', 'quadmath']
        force_lib = []
        compile_args = ['-std=gnu99']
    # e.g. linux-x86_64
    elif platform.startswith('linux'):
        libraries = ['madx', 'stdc++', 'c']
        # DT_RUNPATH is intransitive, i.e. not used for indirect dependencies
        # like 'cpymad -> libmadx -> libptc'. Therefore, on platforms where
        # DT_RUNPATH is used (py35) rather than DT_RPATH (py27) and MAD-X is
        # installed in a non-system location, we need to link against libptc
        # directly to make it discoverable:
        # (an even better solution is to build MAD-X with RPATH=.)
        force_lib = ['ptc']
        compile_args = ['-std=gnu99']
    # Mac/others(?) (e.g. darwin*)
    else:
        libraries = ['madx', 'stdc++', 'c']
        # NOTE: no `force_lib` since Mac's ld doesn't support `--as-needed`:
        force_lib = []
        compile_args = ['-std=gnu99']
    link_args = (['-Wl,--no-as-needed'] +
                 ['-l'+lib for lib in force_lib] +
                 ['-Wl,--as-needed']) if force_lib else []
    # Common arguments for the Cython extensions:
    return dict(
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
            'minrpc>=0.0.5',
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
