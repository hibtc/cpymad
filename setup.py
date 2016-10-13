"""
Installation script for cpymad.

Usage:
    python setup.py install --madxdir=/path/to/madx/installation

For more information, see
    http://hibtc.github.io/cpymad/installation
"""

# Make sure setuptools is available. NOTE: the try/except hack is required to
# make installation work with pip: If an older version of setuptools is
# already imported, `use_setuptools()` will just exit the current process.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, Extension
from distutils.util import get_platform, convert_path
from distutils import sysconfig

import sys
from os import path


# setuptools.Extension automatically converts all '.pyx' extensions to '.c'
# extensions if detecting that neither Cython nor Pyrex is available. Early
# versions of setuptools don't know about Cython. Since we don't use Pyrex
# in this module, this leads to problems in the two cases where Cython is
# available and Pyrex is not or vice versa. Therefore, setuptools.Extension
# needs to be patched to match our needs:
try:
    # Use Cython if available:
    from Cython.Build import cythonize
except ImportError:
    # Otherwise, always use the distributed .c instead of the .pyx file:
    def cythonize(extensions):
        def pyx_to_c(source):
            return source[:-4]+'.c' if source.endswith('.pyx') else source
        for ext in extensions:
            ext.sources = list(map(pyx_to_c, ext.sources))
            missing_sources = [s for s in ext.sources if not path.exists(s)]
            if missing_sources:
                raise OSError(('Missing source file: {0[0]!r}. '
                               'Install Cython to resolve this problem.')
                              .format(missing_sources))
        return extensions
else:
    orig_Extension = Extension
    class Extension(orig_Extension):
        """Extension that *never* replaces '.pyx' by '.c' (using Cython)."""
        def __init__(self, name, sources, *args, **kwargs):
            orig_Extension.__init__(self, name, sources, *args, **kwargs)
            self.sources = sources


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


def get_long_description():
    """Compose a long description for PyPI."""
    long_description = None
    try:
        long_description = read_file('README.rst').decode('utf-8')
        long_description += '\n' + read_file('COPYING.rst').decode('utf-8')
        long_description += '\n' + read_file('CHANGES.rst').decode('utf-8')
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
    if platform.startswith('win'):      # win32/win-amd64:
        libraries = ['madx', 'stdc++', 'ptc', 'gfortran']
    else:
        libraries = ['madx', 'stdc++', 'c']
    # NOTE: we don't want the following on Mac (platform=darwin*), since ld
    # doesn't support ``--as-needed`` there.
    force_lib = []
    if platform.startswith('linux'):    # e.g. linux-x86_64
        # DT_RUNPATH is intransitive, i.e. not used for indirect dependencies
        # like 'cpymad -> libmadx -> libptc'. Therefore, on platforms where
        # DT_RUNPATH is used (py35) rather than DT_RPATH (py27) and MAD-X is
        # installed in a non-system location, we need to link against libptc
        # directly to make it discoverable:
        force_lib = ['ptc']
    if force_lib:
        link_args = (['-Wl,--no-as-needed'] +
                     ['-l'+lib for lib in force_lib] +
                     ['-Wl,--as-needed'])
    else:
        link_args = []
    # Common arguments for the Cython extensions:
    return dict(
        libraries=libraries,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=['-std=gnu99'],
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
            'setuptools',
            'numpy',
            'PyYAML',
            'minrpc>=0.0.3',
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
