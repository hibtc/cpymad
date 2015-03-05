# standard library
import unittest

# utilities
import _compat

# tested objects
from cpymad import util


class TestUtil(unittest.TestCase, _compat.TestCase):

    """Tests for the objects in :mod:`cpymad.util`."""

    def test_is_identifier(self):
        self.assertTrue(util.is_identifier('hdr23_oei'))
        self.assertTrue(util.is_identifier('_hdr23_oei'))
        self.assertFalse(util.is_identifier('2_hdr23_oei'))
        self.assertFalse(util.is_identifier('hdr oei'))
        self.assertFalse(util.is_identifier('hdr@oei'))

    def test_strip_element_suffix(self):
        self.assertEqual(util.strip_element_suffix('foo.23'), 'foo.23')
        self.assertEqual(util.strip_element_suffix('foo.23:43'), 'foo.23')
        # TODO: this should raise:
        self.assertEqual(util.strip_element_suffix('foo:23o'), 'foo:23o')

    def test_add_element_suffix(self):
        self.assertEqual(util.add_element_suffix('foo.23'), 'foo.23:1')
        self.assertEqual(util.add_element_suffix('foo.23:43'), 'foo.23:43')
        # TODO: this should raise:
        self.assertEqual(util.add_element_suffix('foo:23o'), 'foo:23o:1')

    def test_normalize_range_name(self):
        self.assertEqual(util.normalize_range_name('dr:1'), 'dr')
        self.assertEqual(util.normalize_range_name('lebt$end'), '#e')
        self.assertEqual(util.normalize_range_name('dr'), 'dr')

    # TODO: test other functions


if __name__ == '__main__':
    unittest.main()
