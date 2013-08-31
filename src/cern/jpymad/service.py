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
'''
performs all the initialization stuff to access JMad from python.

Created on Nov 11, 2010

@author: kaifox
'''

# py4j for communicating with jmad
from cern.pymad.abc import PyMadService

from modeldef import JPyMadModelDefinition
from model import JPyMadModel
import jmad as jm
import atexit


class JPyMadService(PyMadService):
    """
    the service which is the main facade for the JPyMad - implementation
    """
    def __init__(self, start=None,jmadhome=None, **kwargs):
        super(JPyMadService, self)

        self._started_jmad = False

        # YIL edit: What about this?
        atexit.register(self.cleanup)
        for key, value in kwargs.items():
            print("WARN: unhandled option '" + key + "' for JPyMandService. Ignoring it.")

        if not start is None:
            if start is "gui":
                self._started_jmad = jm.start_gui(jmadhome)
            elif start is "service":
                self._started_jmad = jm.start_pymadservice(jmadhome)
            else:
                print("WARN: unhandled start='" + start + "' for JPymandService (start can be one of 'gui' or 'service'). Ignoring it.")

        self.jmad_service = jm.connect()



    @property
    def mdefs(self):
        model_definition_manager = self.jmad_service.getModelDefinitionManager()
        mdefs = []
        for model_definition in model_definition_manager.getAllModelDefinitions():
            mdefs.append(JPyMadModelDefinition(model_definition))

        return mdefs

    @property
    def mdefnames(self):
        return [mdef.name for mdef in self.mdefs]

    @property
    def models(self):
        model_manager = self.jmad_service.getModelManager()
        return [JPyMadModel(model) for model in model_manager.getModels()]

    def get_mdef(self, name):
        ''' returns the model definition of the given name '''
        model_definition_manager = self.jmad_service.getModelDefinitionManager()
        model_definition = model_definition_manager.getModelDefinition(name)
        return JPyMadModelDefinition(model_definition)

    def create_model(self, mdef):
        if  isinstance(mdef, str):
            model_definition = self.get_mdef(mdef)
            if model_definition == None:
                raise Exception("Model definition '" + mdef + "' not found.")
        else:
            model_definition = mdef

        jmm = self.jmad_service.createModel(model_definition.jmmd)
        jmm.init()
        print(jm.stop)
        return JPyMadModel(jmm)

    def delete_model(self, model):
        self.jmad_service.deleteModel(model.jmm)

    @property
    def am(self):
        """
        retrieves the active model from the model manager
        """
        model_manager = self.jmad_service.getModelManager()
        active_model = model_manager.getActiveModel()
        if active_model == None:
            return None
        else:
            return JPyMadModel(active_model)

    @am.setter
    def am(self, pymadmodel):
        model_manager = self.jmad_service.getModelManager()
        model_manager.setActiveModel(pymadmodel.jmm)

    def cleanup(self):
        if (self._started_jmad == True):
            jm.stop()

