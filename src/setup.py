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

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os,sys
sourcefiles=[["cern/cpymad/madx.pyx"]]
pythonsrc=["cern",
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

libs=['madx', "X11", "z", "pthread", "c", "stdc++"]

def add_dir(directory,dirlist):
    if os.path.isdir(directory):
        if directory not in dirlist:
            dirlist.append(directory)
        

home=os.environ['HOME']
includedirs=[]
libdirs=[]

add_dir(os.path.join(sys.prefix,'include'),includedirs)
add_dir('/usr/local/include',includedirs)
add_dir('/usr/include',includedirs)
add_dir(os.path.join(home,'.local','include'),includedirs)
add_dir('/afs/cern.ch/user/y/ylevinse/.local/include',includedirs)
mad_include_found=False
for f in includedirs:
    if os.path.isdir(os.path.join(f,'madX')):
        mad_include_found=True
if not mad_include_found:
    raise ValueError("Cannot find folder with Mad-X headers")

add_dir(os.path.join(home,'.local','lib'),libdirs)
add_dir(os.path.join(home,'.local','lib64'),libdirs)

mods=[Extension('cern.madx',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1')],
                    include_dirs = includedirs,
                    libraries = libs,
                    sources = sourcefiles[0],
                    library_dirs = libdirs,
                    # The following make sure all
                    # library folders are known to the extension
                    extra_link_args = ['-Wl,-R'+d for d in libdirs]
                    ),
      ]

setup(
    name='PyMAD',
    version='0.1',
    description='Interface to Mad-X, using Cython or Py4J through JMAD',
    cmdclass = {'build_ext': build_ext},
    ext_modules = mods,
    author='PyMAD developers',
    author_email='pymad@cern.ch',
    license = 'CERN Standard Copyright License',
    packages = pythonsrc,
    package_data={'cern.cpymad': cdata},
    )
