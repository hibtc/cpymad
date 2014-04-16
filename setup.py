#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2011, CERN. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from distutils.util import get_platform

import sys
from os import path

# Version of pymad (major,minor):
PYMADVERSION=['0','7']


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
        return extensions
else:
    orig_Extension = Extension
    class Extension(orig_Extension):
        """Extension that *never* replaces '.pyx' by '.c' (using Cython)."""
        def __init__(self, name, sources, *args, **kwargs):
            orig_Extension.__init__(self, name, sources, *args, **kwargs)
            self.sources = sources

# Subclass the build_ext command for building C-extensions. This enables to
# use the ``setup_requires`` argument of setuptools to install a missing
# numpy dependency, before having to know the location of the numpy header
# files. All in all, this should make the setup more properly bootstrapped.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Before importing numpy, we need to make sure it doesn't think it
        # is still during its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        # Add location of numpy headers:
        self.include_dirs.append(numpy.get_include())

# Parse command line option: --madxdir=/path/to/madxinstallation. We could
# use build_ext.user_options instead, but then the --madxdir argument can
# be passed only to the 'build_ext' command, not to 'build' or 'install',
# which is a minor nuisance.
for arg in sys.argv:
    if arg.startswith('--madxdir='):
        sys.argv.remove(arg)
        prefix = arg.split('=', 1)[1]
        include_dirs = [path.join(prefix, 'include')]
        lib_path_candidates = [path.join(prefix, 'lib'),
                               path.join(prefix, 'lib64')]
        library_dirs = list(filter(path.isdir, lib_path_candidates))
        break
else:
    # Let's just use the default system headers:
    include_dirs = library_dirs = []

# required libraries
if get_platform() == "win32":
    libraries = ['madx', 'stdc++', 'ptc', 'gfortran', 'msvcrt']
else:
    libraries = ['madx', 'stdc++', 'c']

# Common arguments for the Cython extensions:
extension_args = dict(
    define_macros=[('MAJOR_VERSION', PYMADVERSION[0]),
                   ('MINOR_VERSION', PYMADVERSION[1])],
    libraries=libraries,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=library_dirs)

# Compose a long description for PyPI:
long_description = None
try:
    long_description = open('README.rst').read()
    long_description += '\n' + open('CHANGES.rst').read()
except IOError:
    pass

setup(
    name='cern-pymad',
    version='.'.join(map(str, PYMADVERSION)),
    description='Interface to Mad-X, using Cython or Py4J through JMAD',
    long_description=long_description,
    url='http://cern.ch/pymad',
    package_dir={'':'src'},
    cmdclass={'build_ext':build_ext},
    ext_modules = cythonize([
        Extension('cern.cpymad.libmadx',
                  sources=["src/cern/cpymad/libmadx.pyx"],
                  **extension_args),
    ]),
    namespace_packages=[
        'cern'
    ],
    packages = [
        "cern",
        "cern.resource",
        "cern.cpymad",
        "cern.cpymad._couch",
        "cern.cpymad._connection",
        "cern.jpymad",
        "cern.jpymad.tools",
        "cern.pymad",
        "cern.pymad.io",
        "cern.pymad.abc",
        "cern.pymad.domain"
    ],
    include_package_data=True, # include files matched by MANIFEST.in
    author='PyMAD developers',
    author_email='pymad@cern.ch',
    setup_requires=['numpy'],
    install_requires=['numpy', 'PyYAML'],
    license = 'CERN Standard Copyright License'
)

