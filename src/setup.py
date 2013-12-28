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

import sys
from os import path

# Version of pymad (major,minor):
PYMADVERSION=['0','4']

# With Python 2.6, the cythoning does not work with setuptools:
if sys.version_info < (2,7):
    from distutils.core import setup
    from distutils.extension import Extension
else: # should be default asap
    from setuptools import setup
    from setuptools.extension import Extension

try:
    long_description = open('../README.rst').read()
except IOError:
    long_description = None

from Cython.Distutils import build_ext
import platform
from distutils.util import get_platform
import numpy

# parse command line option: --madxdir=/path/to/madxinstallation
special_madxdir = ''
for arg in list(sys.argv):  # avoid problems due to side-effects by copying sys.argv into a temporary list
    if arg.startswith('--madxdir='):
        special_madxdir = arg.split('=', maxsplit=1)[1]
        sys.argv.remove(arg)

def add_dir(dirlist, directory):
    if path.isdir(directory) and directory not in dirlist:
        dirlist.append(directory)

# guess prefixes for possible header/library locations
if special_madxdir:
    _prefixdirs = [special_madxdir]
else:
    _prefixdirs = [sys.prefix]
add_dir(_prefixdirs, '/usr')
add_dir(_prefixdirs, '/usr/local')
add_dir(_prefixdirs, path.join(path.expanduser('~'),'.local'))

# extra include pathes: madx
includedirs = [path.join(d, 'include')
               for d in _prefixdirs
               if path.isdir(path.join(d, 'include', 'madX'))]
if not includedirs:
    raise RuntimeError("Cannot find folder with Mad-X headers")

# Add numpy include directory (for cern.libmadx.table):
includedirs.append(numpy.get_include())

# static library pathes
libdirs = []        # static library pathes
for prefixdir in _prefixdirs:
    add_dir(libdirs, path.join(prefixdir,'lib'))
    if platform.architecture()[0]=='64bit':
        add_dir(libdirs, path.join(prefixdir,'lib64'))

# runtime library pathes
rlibdirs = []
for ldir in libdirs:
    if any(path.isfile(path.join(ldir,'libmadx.'+suffix))
           for suffix in ['so','dll','dylib']):
        rlibdirs = [ldir]
        libdirs = [ldir]    # overwrites libdirs, is this intentional?
        break

# required libraries
libs = ['madx', 'stdc++']
if get_platform() == "win32":
    libs += ['ptc', 'gfortran', 'msvcrt']
else:
    libs += ['c']

# common cython arguments
cython_args = dict(
    define_macros=[('MAJOR_VERSION', PYMADVERSION[0]),
                   ('MINOR_VERSION', PYMADVERSION[1])],
    include_dirs=includedirs,
    libraries=libs,
    library_dirs=libdirs,
    runtime_library_dirs=rlibdirs)

setup(
    name='cern-pymad',
    version='.'.join(map(str, PYMADVERSION)),
    description='Interface to Mad-X, using Cython or Py4J through JMAD',
    long_description=long_description,
    url='http://cern.ch/pymad',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension('cern.madx',
                  sources=["cern/madx.pyx"],
                  **cython_args),
        Extension('cern.libmadx.table',
                  sources=["cern/libmadx/table.pyx"],
                  **cython_args),
    ],
    packages = [
        "cern",
        "cern.libmadx",
        "cern.resource",
        "cern.cpymad",
        "cern.cpymad._couch",
        "cern.jpymad",
        "cern.jpymad.tools",
        "cern.pymad",
        "cern.pymad.io",
        "cern.pymad.abc",
        "cern.pymad.tools",
        "cern.pymad.domain"
    ],
    include_package_data=True, # include files matched by MANIFEST.in
    author='PyMAD developers',
    author_email='pymad@cern.ch',
    license = 'CERN Standard Copyright License')

