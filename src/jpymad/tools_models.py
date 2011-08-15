# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:21:25 2010

@author: kaifox
"""

from service import pms
from model import PyMadModel

def ls_mdefs():
    """
    lists all the currently available model definitions
    """
    model_definition_manager = pms.jmad_service.getModelDefinitionManager()
    
    print "Available model definitions:"
    print "----------------------------"
    for model_definition in model_definition_manager.getAllModelDefinitions():
        print model_definition.getName()

def get_models():
    """
    convenience method to get all models
    """
    model_manager = pms.jmad_service.getModelManager()
    return [PyMadModel(model) for model in model_manager.getModels()]

def ls_models():
    """
    lists all the currently created models
    """
    print "Model instances:"
    print "----------------"
    for model in get_models():
        print model.getName()
        
def create_model(model_definition_name):
    """
    retrieves the model definition of the given name from the service and creates the model
    """
    model_definition_manager = pms.jmad_service.getModelDefinitionManager()

    model_definition = model_definition_manager.getModelDefinition(model_definition_name)
    if model_definition == None:
        raise Exception("Model definition '" + model_definition_name + "' not found.")

    jmm = pms.jmad_service.createModel(model_definition)
    jmm.init()
    return PyMadModel(jmm)
    
def am():
    """
    retrieves the active model from the model manager
    """
    model_manager = pms.jmad_service.getModelManager()
    active_model = model_manager.getActiveModel()
    if active_model == None:
        return None
    else:
        return PyMadModel(active_model)

def set_am(pymadmodel):
    model_manager = pms.jmad_service.getModelManager()
    model_manager.setActiveModel(pymadmodel.jmm)
