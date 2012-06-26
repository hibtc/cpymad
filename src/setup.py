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

if "bdist_egg" in sys.argv:
    from setuptools import setup
else:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform

# ugly hack to add --madxdir=/path/to/madxinstallation
special_madxdir=''
for arg in sys.argv:
    if '--madxdir=' in arg:
        special_madxdir=arg.split('=')[1]
        sys.argv.remove(arg)


sourcefiles=[["cern/madx.pyx"],["cern/libmadx/table.pyx"]]
pythonsrc=["cern",
           "cern.libmadx",
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
    for end in ['.madx','.str','.seq','.tfs']:
        cdata.append('_models/re*data'+'/*'*j+end)

libs=['madx', "c", "stdc++"]

def add_dir(directory,dirlist):
    if os.path.isdir(directory):
        if directory not in dirlist:
            dirlist.append(directory)


home=os.environ['HOME']
includedirs=['/usr/lib/python2.7/site-packages/numpy/core/include/']
libdirs=[]
rlibdirs=[]
if special_madxdir:
    _prefixdirs=[special_madxdir]
else: # making some guesses...
    _prefixdirs=[
        sys.prefix,
        ]
for prefixdir in ['/usr',
        '/usr/local',
        os.path.join(home,'.local'),
        '/afs/cern.ch/user/y/ylevinse/.local']:
    add_dir(prefixdir,_prefixdirs)

for prefixdir in _prefixdirs:
    if os.path.isdir(os.path.join(prefixdir,'include','madX')):
        add_dir(os.path.join(prefixdir,'include'),includedirs)
        break
if not includedirs:
    raise ValueError("Cannot find folder with Mad-X headers")

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
_modver=['0','2']
mods=[Extension('cern.madx',
        define_macros = [('MAJOR_VERSION', _modver[0]),
                            ('MINOR_VERSION', _modver[1])],
        include_dirs = includedirs,
        libraries = libs,
        sources = sourcefiles[0],
        library_dirs = libdirs,
        runtime_library_dirs= rlibdirs
        ),
      Extension('cern.libmadx.table',
                    define_macros = [('MAJOR_VERSION', _modver[0]),
                                     ('MINOR_VERSION', _modver[1])],
                    include_dirs = includedirs,
                    libraries = libs,
                    sources = sourcefiles[1],
                    library_dirs = libdirs,
                    runtime_library_dirs= rlibdirs
                    ),
      ]

setup(
    name='PyMAD',
    version='0.2',
    description='Interface to Mad-X, using Cython or Py4J through JMAD',
    cmdclass = {'build_ext': build_ext},
    ext_modules = mods,
    author='PyMAD developers',
    author_email='pymad@cern.ch',
    license = 'CERN Standard Copyright License',
    packages = pythonsrc,
    package_data={'cern.cpymad': cdata},
    )
