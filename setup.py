"""
Installation script for CPyMAD.

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

# Let's just use the default system headers:
include_dirs = []
library_dirs = []

# Parse command line option: --madxdir=/path/to/madxinstallation. We could
# use build_ext.user_options instead, but then the --madxdir argument can
# be passed only to the 'build_ext' command, not to 'build' or 'install',
# which is a minor nuisance.
for arg in sys.argv[:]:
    if arg.startswith('--madxdir='):
        sys.argv.remove(arg)
        prefix = path.expanduser(arg.split('=', 1)[1])
        lib_path_candidates = [path.join(prefix, 'lib'),
                               path.join(prefix, 'lib64')]
        include_dirs += [path.join(prefix, 'include')]
        library_dirs += list(filter(path.isdir, lib_path_candidates))

# required libraries
if get_platform() == "win32" or get_platform() == "win-amd64":
    libraries = ['madx', 'stdc++', 'ptc', 'gfortran', 'msvcrt']
else:
    libraries = ['madx', 'stdc++', 'c']

# Common arguments for the Cython extensions:
extension_args = dict(
    libraries=libraries,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=library_dirs,
    extra_compile_args=['-std=c99'],
)

# Compose a long description for PyPI:
long_description = None
try:
    long_description = open('README.rst').read()
    long_description += '\n' + open('COPYING.rst').read()
    long_description += '\n' + open('CHANGES.rst').read()
except IOError:
    pass

# read metadata from cpymad/__init__.py
def exec_file(path):
    """Execute a python file and return the `globals` dictionary."""
    namespace = {}
    with open(convert_path(path)) as f:
        exec(f.read(), namespace, namespace)
    return namespace

metadata = exec_file('cpymad/__init__.py')

setup(
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

    ext_modules = cythonize([
        Extension('cpymad.libmadx',
                  sources=["cpymad/libmadx.pyx"],
                  **extension_args),
    ]),
    packages = [
        "cpymad",
        "cpymad.resource",
    ],
    include_package_data=True, # include files matched by MANIFEST.in
    install_requires=[
        'setuptools',
        'numpy',
        'PyYAML',
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'License :: OSI Approved :: Apache Software License',
        'License :: Other/Proprietary License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

