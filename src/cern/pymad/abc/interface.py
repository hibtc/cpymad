# encoding: utf-8
#-------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2013, CERN. All rights reserved.
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
'''
.. module: cern.abc.pymad.interface

Created on 01 Sep 2013

.. moduleauthor:: Thomas Gläßle <t_glaessle@gmx.de>

'''

from abc import ABCMeta

Interface = ABCMeta('Interface', (object,), {})

Interface.__doc__ = '''
Base class for abstract classes.

This class serves as a  compatibility layer between python2 and python3.
It  enables  code  to   use  `metaclass=ABCMeta`  across  both  language
versions.

'''

