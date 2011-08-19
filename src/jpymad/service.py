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
from jpymad.jmad import connect, start_gui, start_pymadservice, stop
'''
performs all the initialization stuff to access JMad from python.

Created on Nov 11, 2010

@author: kaifox
'''

# py4j for communicating with jmad
from pymad.abc import PyMadService

from jpymad.modeldef import JPyMadModelDefinition
from jpymad.model import JPyMadModel


class JPyMadService(PyMadService):
    """
    the service which is the main facade for the JPyMad - implementation
    """
    def __init__(self, **kwargs):
        super(JPyMadService, self)
        
        start = None
        jmadhome = None
        for key, value in kwargs.items():
            if key is 'start':
                start = value
            elif key is 'jmadhome':
                jmadhome = value
            else:
                print "WARN: unhandled option '" + key + "' for JPyMandService. Ignoring it." 
       
        if not start is None: 
            if start is "gui":
                start_gui(jmadhome)
            elif start is "service":
                start_pymadservice(jmadhome)
            else:
                print "WARN: unhandled start='" + start + "' for JPymandService (start can be one of 'gui' or 'service'). Ignoring it."
            
        self.jmad_service = connect()
        
    
    
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
        
    def stop(self):
        stop()



    
    
        

    
   
    

