# encoding: utf-8
"""
Tests for the model.Locator functionality.
"""

# tested classes
from cern.cpymad.model import Locator

# test utilities
import unittest
import os
from cern.resource.file import FileResource

__all__ = [
    'TestLocator',
]


class TestLocator(unittest.TestCase):

    """
    Test the Locator (standard filesystem locator).
    """

    unicode_data = u"€®æة《±∓"

    def setUp(self):
        here = os.path.dirname(__file__)
        self.base = os.path.join(here, 'data')
        self.locator = Locator(FileResource(self.base))

    def test_list_models(self):
        """Verify the results of the list_models method."""
        self.assertEqual(
            set(self.locator.list_models()),
            set(('a', 'b', 'c', 'd', 'e', 'lebt')))

    def test_get(self):
        """Test that ValueError is raised iff model does not exist."""
        # cannot get non-existing model:
        self.assertRaises(ValueError, self.locator.get_definition, 'f')
        # can get existing real models:
        self.locator.get_definition('a')
        self.locator.get_definition('b')
        self.locator.get_definition('c')
        self.locator.get_definition('d')
        self.locator.get_definition('e')
        self.locator.get_definition('lebt')

    def test_encoding(self):
        """Test that the model is loaded with the correct encoding."""
        b = self.locator.get_definition('b')
        self.assertEqual(b['unicode'],
                         self.unicode_data)

    def test_mro(self):
        """Test the resolution order when using 'extends'."""
        b = self.locator.get_definition('b')['data']
        c = self.locator.get_definition('c')['data']
        # overwritten properties:
        self.assertEqual((b['b'], b['ab'], b['bc'], b['abc']),
                         ('b', 'b', 'b', 'b'))
        self.assertEqual((c['c'], c['ac'], c['bc'], c['abc']),
                         ('c', 'c', 'c', 'c'))
        # inherited properties:
        self.assertEqual(b['a'], 'a')
        self.assertEqual(c['a'], 'a')
        self.assertEqual(c['ab'], 'b')
        self.assertEqual(c['b'], 'b')

    def test_path_resolution(self):
        """Test the .get and .get_by_dict methods of ModelData."""
        b = self.locator.get_definition('b')
        c = self.locator.get_definition('c')
        bf = b['files']
        cf = c['files']
        br = self.locator.get_repository(b)
        cr = self.locator.get_repository(c)
        # the first file is defined in ?/a.yml
        self.assertEqual(br.get(bf[0]).yaml()['path'],
                         'b/a.yml')
        self.assertEqual(cr.get(cf[0]).yaml()['path'],
                         'c/a.yml')
        # second file: ?/b.yml
        self.assertEqual(br.get(bf[1]).yaml()['path'],
                         'b/b.yml')
        self.assertEqual(cr.get(cf[1]).yaml()['path'],
                         'c/b.yml')
        # third file: ?/c.yml
        self.assertEqual(cr.get(cf[2]).yaml()['path'],
                         'c/c.yml')


if __name__ == '__main__':
    unittest.main()
