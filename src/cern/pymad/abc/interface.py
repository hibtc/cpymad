
__all__ = ['ABCMetaDoc', 'abstractmethod']

from abc import ABCMeta, abstractmethod
from types import FunctionType
from inspect import getdoc


class ABCMetaDoc(ABCMeta):
    """
    Metaclass used to automatically inherit docstrings from base classes.
    """
    def __new__(meta, classname, bases, classDict):
        """
        Create new class and copy all docstrings from base classes.
        """
        cls = ABCMeta.__new__(meta, classname, bases, classDict)
        for name in classDict:
            fn = classDict[name]
            if type(fn) != FunctionType:
                continue
            docs = []
            for base in cls.mro():
                if not hasattr(base, name) or base is object:
                    continue
                basefn = getattr(base, name)
                basedoc = getdoc(basefn)
                if basedoc:
                    docs.append((base, basedoc))
            if len(docs) == 0:
                doc = None
            elif len(docs) == 1:
                doc = docs[0][1]
            else:
                doc = ""
                if docs[0][0] is cls:
                    doc += docs[0][1]
                    docs = docs[1:]
                for d in docs:
                    doc += "\n\nOverrides %s.%s" % (d[0].__name__, name)
                    doc += d[1]
                doc = doc.lstrip('\n')
            cls.__dict__[name].__doc__ = doc
        return cls

def plain(*args, **kwargs):
    pass

Interface = ABCMetaDoc('Interface', (object,), {'__init__': plain})

