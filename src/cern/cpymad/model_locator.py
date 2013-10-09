# -*- coding: iso-8859-15 -*-
#------------------------------------------------------------------------------
# This file is part of PyMad.
#
# Copyright (c) 2013, Thomas Gläßle. All rights reserved.
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
#------------------------------------------------------------------------------
"""
Contains classes used locate and load models from resource locations.

This might be fully integratable into cpymad.service.

"""
__all__ = [
    'ModelData',
    'ModelLocator',
    'MergedModelLocator',
    'DistinctModelLocator',
    'ChainModelLocator'
]

from cern.pymad.abc.interface import Interface, abstractmethod
from collections import Mapping, OrderedDict
from itertools import chain
from cern.resource.file import FileResource
import os.path

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list) and k in d:
            d[k].extend(v)
        else:
            d[k] = v
    return d

class ModelData(object):
    """
    Loader for individual data objects from a model resource collection.

    Has three public fields: model, resource, repository. The former is a
    dictionary containing the fully expanded model definition. The latter
    two are ResourceProvider instances able to load actual data for
    repository/resource data respectively. 

    """
    __slots__ = ['name', 'model', 'resource', 'repository']

    def __init__(self, name, model, resource, repository):
        """
        Initialize the ModelData instance.
        """
        self.name = name
        self.model = model
        self.resource = resource
        self.repository = repository

    def get_by_dict(self, file_dict):
        """
        Get a ResourceProvider object for the given file.

        file_dict is a mapping in the form found in model definitions,
        containing a 'location' and a 'path' key.

        """
        return self.get(file_dict['path'],
                        file_dict.get('location', 'repository'))

    def get(self, filename, kind='repository'):
        """
        Get a ResourceProvider object for the given file.

        :param string filename: name of the resource
        :param string kind: one of ('repository', 'resource')

        """
        if kind.lower() == 'repository':
            return self.repository.get(filename)
        elif kind.lower() == 'resource':
            return self.resource.get(filename)
        else:
            raise ValueError("Invalid resource kind: %s" % kind)


class ModelLocator(Interface):
    """
    Model locator and loader base class.

    Serves the purpose of locating models and returning corresponding
    resource providers.

    """
    def list_models(self):
        """
        Iterate all available models.

        Returns an iterable that may be a generator object.

        """
        pass

    def get_model(self, name):
        """
        Get the first found model with the specified name.

        Returns a ModelData instance.
        Raises a ValueError if no model with the given name is found.

        """
        pass


class MergedModelLocator(ModelLocator):
    """
    Model locator for json files that contain multiple model definitions.

    These are the model definition files that are currently used by default
    for filesystem resources.

    """
    def __init__(self, resource_provider):
        """
        Initialize a merged model locator instance.

        The resource_provider parameter must be a ResourceProvider instance
        that points to the filesystem location where the .cpymad.json model
        files are stored.

        """
        self.res_provider = resource_provider

    def list_models(self):
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.json'):
            mdefs = self.res_provider.json(res_name)
            for n,d in mdefs.items():
                if d['real']:
                    yield n

        #----------------------------------------
        # return chain.from_iterable(
        #     name for name,mdef in (
        #         self.res_provider.json(res).items()
        #         for res in self.res_provider.listdir_filter(
        #             ext='.cpymad.json')
        #     ) if mdef['real'])
        #----------------------------------------

    def get_model(self, name):
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.json'):
            mdefs = self.res_provider.json(res_name)
            # restrict only to 'real' models (don't do that?):
            if name in (n for n,d in mdefs.items() if d['real']):
                break
        else:
            raise ValueError("The model you asked for does not exist in the database")

        # TODO: in previous implementation the bases did overwrite
        # specialized version. was this intended?

        # expand the model using its bases specified in 'extends'. try to
        # provide a useful MRO:
        mro = OrderedDict()
        more = [name]
        while more:
            even_more = []
            for base in more:
                if base not in mro:
                    mro[base] = mdef = mdefs[base]
                    even_more.extend(mdef.get('extends', []))
            more = even_more

        # TODO: this could be done using some sort of ChainMap, i.e.
        # merging at lookup time instead of at creation time. but this is
        # probably not worth the trouble for now.
        real_mdef = {}
        for virt_mdef in reversed(mro.values()):
            deep_update(real_mdef, virt_mdef)

        # instantiate the resource providers for model resource data
        res_offs = real_mdef['path-offsets']['resource-offset']
        repo_offs = real_mdef['path-offsets']['repository-offset']
        res_prov = self.res_provider.get('resdata').get(res_offs)

        # the repository location may be overwritten by dbdirs:
        for dbdir in real_mdef.get('dbdirs', []):
            if os.path.isdir(dbdir):
                repo_prov = FileResource(os.path.join(dbdir, repo_offs))
                break
        else:
            repo_prov = self.res_provider.get('repdata').get(repo_offs)

        # return a wrapper for the modeldata
        return ModelData(name,
                         real_mdef,
                         resource=res_prov,
                         repository=repo_prov)



class DistinctModelLocator(ModelLocator):
    """
    Model locator for a resource provider that handles each model distinctly.

    This behaviour is found in couchdb model resources.

    """
    def __init__(self, resource_provider):
        """Initialize using a ResourceProvider."""
        self.resource_provider = resource_provider

    def list_models(self):
        return self.resource_provider.listdir()

    def get_model(self, name):
        res = self.resource_provider.get(name)
        mdef = res.json()
        res_prov = res.get(mdef['path-offsets']['resource-offset'])
        repo_prov = res.get(mdef['path-offsets']['repository-offset'])
        return ModelData(name,
                         mdef,
                         resource=res_prov,
                         repository=repo_prov)



class ChainModelLocator(ModelLocator):
    """
    Chain multiple model locators.
    """
    def __init__(self):
        """Initialize empty chain."""
        self._locators = []

    def add_locator(self, locator):
        """Append (chain) a ModelLocator."""
        self._locators.append(locator)

    def list_models(self):
        return chain.from_iterable(locator.list_models()
                                   for locator in self._locators)

    def get_model(self, name):
        for locator in self._locators:
            try:
                return locator.get_model(name)
            except ValueError:
                pass
        else:
            raise ValueError("Model not found: %s" % name)

