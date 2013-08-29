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
def runj():
    import jpymad
    run(jpymad)
def runc():
    import cpymad
    run(cpymad)

def run(pymad):
    l=pymad.model('lhc')
    print("Available sequences: "+str(l.list_sequences()))
    # would it be possible to implement
    # same type of return here?
    t,p=l.twiss('lhcb1')

if __name__=="__main__":
    #runj()
    runc()
