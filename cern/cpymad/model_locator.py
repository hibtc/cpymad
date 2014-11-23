# encoding: utf-8
"""
Contains classes used locate and load models from resource locations.

This might be fully integratable into cpymad.service.

"""
__all__ = [
    'ModelData',
    'ModelLocator',
    'MergedModelLocator',
]

from collections import Mapping
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

def C3_mro(get_bases, *bases):
    """
    Calculate the C3 MRO of bases.

    Suppose you intended creating a class K with the given base classes. This
    function returns the MRO which K would have, *excluding* K itself (since
    it doesn't yet exist), as if you had actually created the class.

    Another way of looking at this, if you pass a single class K, this will
    return the linearization of K (the MRO of K, *including* itself).

    http://code.activestate.com/recipes/577748-calculate-the-mro-of-a-class/

    """
    seqs = [[C] + C3_mro(get_bases, *get_bases(C)) for C in bases] + [list(bases)]
    result = []
    while True:
      seqs = list(filter(None, seqs))
      if not seqs:
          return result
      try:
          head = next(seq[0] for seq in seqs
                      if not any(seq[0] in s[1:] for s in seqs))
      except StopIteration:
          raise TypeError("inconsistent hierarchy, no C3 MRO is possible")
      result.append(head)
      for seq in seqs:
          if seq[0] == head:
              del seq[0]


class ModelData(object):
    """
    Loader for individual data objects from a model resource collection.

    Has four public fields: name, model, resource, repository. `model` is a
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


class ModelLocator(object):
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
        raise NotImplementedError("ModelLocator.list_models")

    def get_model(self, name, encoding='utf-8'):
        """
        Get the first found model with the specified name.

        Returns a ModelData instance.
        Raises a ValueError if no model with the given name is found.

        """
        raise NotImplementedError("ModelLocator.get_model")


class MergedModelLocator(ModelLocator):
    """
    Model locator for yaml files that contain multiple model definitions.

    These are the model definition files that are currently used by default
    for filesystem resources.

    """
    def __init__(self, resource_provider):
        """
        Initialize a merged model locator instance.

        The resource_provider parameter must be a ResourceProvider instance
        that points to the filesystem location where the .cpymad.yml model
        files are stored.

        """
        self.res_provider = resource_provider

    def list_models(self, encoding='utf-8'):
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.yml'):
            mdefs = self.res_provider.yaml(res_name, encoding=encoding)
            for n,d in mdefs.items():
                if d['real']:
                    yield n

    def get_model(self, name, encoding='utf-8'):
        for res_name in self.res_provider.listdir_filter(ext='.cpymad.yml'):
            mdefs = self.res_provider.yaml(res_name, encoding=encoding)
            # restrict only to 'real' models (don't do that?):
            if name in (n for n,d in mdefs.items() if d['real']):
                break
        else:
            raise ValueError("The model "+name+" does not exist in the database")

        # expand the model using its bases specified in 'extends'. try to
        # provide a useful MRO:
        def get_bases(model_name):
            return mdefs[model_name].get('extends', [])
        mro = C3_mro(get_bases, name)

        # TODO: this could be done using some sort of ChainMap, i.e.
        # merging at lookup time instead of at creation time. but this is
        # probably not worth the trouble for now.
        real_mdef = {}
        for base in reversed(mro):
            deep_update(real_mdef, mdefs[base])

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
