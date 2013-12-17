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
#!/usr/bin/python

import os,sys

# Version of pymad (major,minor):
PYMADVERSION=['0','4']

# With Python 2.6, the cythoning does not work
# with setuptools:
if sys.version_info < (2,7):
    from distutils.core import setup
    from distutils.extension import Extension
else: # should be default asap
    from setuptools import setup
    from setuptools.extension import Extension

from Cython.Distutils import build_ext
import platform
from distutils.util import get_platform
import numpy

# ugly hack to add --madxdir=/path/to/madxinstallation
special_madxdir=''
for arg in sys.argv:
    if '--madxdir=' in arg:
        special_madxdir=arg.split('=')[1]
        sys.argv.remove(arg)


sourcefiles=[["cern/madx.pyx"],["cern/libmadx/table.pyx"]]
pythonsrc=["cern",
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
           "cern.pymad.domain"]

# list of data files to include..
cdata=['_models/*.json']

# add this to include data array
for j in range(2,10):
    for end in ['.madx','.str','.seq','.tfs', '.xsifx', 'CLICx' ,'.ind92']:
        cdata.append('_models/re*data'+'/*'*j+end)

libs=['madx', 'stdc++']
if get_platform() == "win32":
    libs += ['ptc', 'gfortran', 'msvcrt']
else:
    libs += ['c']

def add_dir(directory,dirlist):
    if os.path.isdir(directory):
        if directory not in dirlist:
            dirlist.append(directory)

includedirs=[]
libdirs=[]
rlibdirs=[]
if special_madxdir:
    _prefixdirs=[special_madxdir]
else: # making some guesses...
    _prefixdirs=[ sys.prefix, ]

for prefixdir in ['/usr',
        '/usr/local',
        os.path.join(os.path.expanduser('~'),'.local')]:
    add_dir(prefixdir,_prefixdirs)

for prefixdir in _prefixdirs:
    if os.path.isdir(os.path.join(prefixdir,'include','madX')):
        add_dir(os.path.join(prefixdir,'include'),includedirs)
        break
if not includedirs:
    raise ValueError("Cannot find folder with Mad-X headers")

# Add numpy include directory (for cern.libmadx.table):
includedirs.append(numpy.get_include())

for prefixdir in _prefixdirs:
    add_dir(os.path.join(prefixdir,'lib'),libdirs)
    if platform.architecture()[0]=='64bit':
        add_dir(os.path.join(prefixdir,'lib64'),libdirs)

for ldir in libdirs:
    if rlibdirs:
        break
    for suffix in ['so','dll','dylib']:
        if os.path.isfile(os.path.join(ldir,'libmadx.'+suffix)):
            rlibdirs=[ldir]
            libdirs=[ldir]
            break
mods=[Extension('cern.madx',
        define_macros = [('MAJOR_VERSION', PYMADVERSION[0]),
                            ('MINOR_VERSION', PYMADVERSION[1])],
        include_dirs = includedirs,
        libraries = libs,
        sources = sourcefiles[0],
        library_dirs = libdirs,
        runtime_library_dirs= rlibdirs
        ),
      Extension('cern.libmadx.table',
                    define_macros = [('MAJOR_VERSION', PYMADVERSION[0]),
                                     ('MINOR_VERSION', PYMADVERSION[1])],
                    include_dirs = includedirs,
                    libraries = libs,
                    sources = sourcefiles[1],
                    library_dirs = libdirs,
                    runtime_library_dirs= rlibdirs
                    ),
      ]

if sys.version_info < (2,7):
    # can be deleted when we don't support Python 2.6 anymore..
    setup(
        name='PyMAD',
        version='.'.join([str(i) for i in PYMADVERSION]),
        description='Interface to Mad-X, using Cython or Py4J through JMAD',
        cmdclass = {'build_ext': build_ext},
        ext_modules = mods,
        author='PyMAD developers',
        author_email='pymad@cern.ch',
        license = 'CERN Standard Copyright License',
        packages = pythonsrc,
        package_data={'cern.cpymad': cdata}
        )
else:
    setup(
        name='PyMAD',
        version='.'.join([str(i) for i in PYMADVERSION]),
        description='Interface to Mad-X, using Cython or Py4J through JMAD',
        cmdclass = {'build_ext': build_ext},
        ext_modules = mods,
        author='PyMAD developers',
        author_email='pymad@cern.ch',
        license = 'CERN Standard Copyright License',
        packages = pythonsrc,
        package_data={'cern.cpymad': cdata},
        setup_requires=['numpy', 'Cython'],
        install_requires=['numpy'],
        )
