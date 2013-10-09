
# tested classes
from cern.pymad.abc.interface import Interface

# test utilities
import unittest
from inspect import getdoc
from abc import abstractmethod

#
class AbstractBase(Interface):
    @abstractmethod
    def foo(self):
        """
        Dummy documentation for AbstractBase.foo
        """
        pass

    @abstractmethod
    def bar(self):
        """
        Dummy documentation for AbstractBase.bar
        """
        pass

def _bar(self):
    """
    Dummy documentation for Derived.bar
    """
    pass

class Derived(AbstractBase):
    def foo(self):
        pass

    bar = _bar

class TestInterface(unittest.TestCase):
    def test_method_doc(self):
        self.assertEqual(
                getdoc(Derived.foo),
                getdoc(AbstractBase.foo))
        self.assertTrue(
                getdoc(Derived.bar).startswith(getdoc(_bar)))
        self.assertTrue(
                getdoc(Derived.bar).endswith(getdoc(AbstractBase.bar)))

