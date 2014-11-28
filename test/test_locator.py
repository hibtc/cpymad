# encoding: utf-8
"""
Tests for the model.Locator functionality.
"""

# tested classes
from cpymad.model import Locator
from cpymad.resource.file import FileResource

# test utilities
import unittest
import os

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
            set(['lebt']))

    def test_get(self):
        """Test that ValueError is raised iff model does not exist."""
        # cannot get non-existing model:
        self.assertRaises(ValueError, self.locator.get_definition, 'f')
        # can get existing real models:
        self.locator.get_definition('lebt')

    def test_encoding(self):
        """Test that the model is loaded with the correct encoding."""
        b = self.locator.get_definition('lebt')
        self.assertEqual(b['unicode'],
                         self.unicode_data)

    def test_path_resolution(self):
        """Test the .get and .get_by_dict methods of ModelData."""
        mdef = self.locator.get_definition('lebt')
        files = mdef['files']
        repo = self.locator.get_repository(mdef)
        self.assertEqual(repo.get(files[0]).yaml()['path'], 'lebt/misc.yml')


if __name__ == '__main__':
    unittest.main()
