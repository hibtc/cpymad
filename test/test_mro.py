# encoding: utf-8
"""
Tests for C3_mro func from cern.cpymad.util.
"""

# tested classes
from cern.cpymad.util import C3_mro

# test utilities
import unittest


__all__ = [
    'TestMROFunc',
]


class TestMROFunc(unittest.TestCase):

    """
    Test the C3_mro() function.

    The function is tested for agreement with standard python MRO.

    Note: the tests will only succeed for versions of python where C3 is
    used, i.e. python>=2.3.
    """

    def mro(self, *bases):
        return tuple(C3_mro(lambda cls: cls.__bases__, *bases))

    def test_SeriousOrderDisagreement(self):
        O = object
        class X(O): pass
        class Y(O): pass
        class A(X, Y): pass
        class B(Y, X): pass
        bases = (A, B)
        self.assertRaises(TypeError, self.mro, A, B)

    def test_TrivialSingleInheritance(self):
        O = object
        class A(O): pass
        class B(A): pass
        class C(B): pass
        class D(C): pass
        self.assertEqual(self.mro(D), D.__mro__)

    def test_Example1(self):
        O = object
        class F(O): pass
        class E(O): pass
        class D(O): pass
        class C(D, F): pass
        class B(D, E): pass
        class A(B, C): pass
        self.assertEqual(self.mro(A), A.__mro__)

    def test_Example2(self):
        O = object
        class F(O): pass
        class E(O): pass
        class D(O): pass
        class C(D, F): pass
        class B(E, D): pass
        class A(B, C): pass
        self.assertEqual(self.mro(A), A.__mro__)

    def test_Example3(self):
        O = object
        class A(O): pass
        class B(O): pass
        class C(O): pass
        class D(O): pass
        class E(O): pass
        class K1(A, B, C): pass
        class K2(D, B, E): pass
        class K3(D, A): pass
        class Z(K1, K2, K3): pass
        self.assertEqual(self.mro(A), A.__mro__)
        self.assertEqual(self.mro(B), B.__mro__)
        self.assertEqual(self.mro(C), C.__mro__)
        self.assertEqual(self.mro(D), D.__mro__)
        self.assertEqual(self.mro(E), E.__mro__)
        self.assertEqual(self.mro(K1), K1.__mro__)
        self.assertEqual(self.mro(K2), K2.__mro__)
        self.assertEqual(self.mro(K3), K3.__mro__)
        self.assertEqual(self.mro(Z), Z.__mro__)


if __name__ == '__main__':
    unittest.main()
