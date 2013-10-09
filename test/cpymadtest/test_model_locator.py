"""
Tests for the classes defined in cern.cpymad.model_locator.
"""
__all__ = ['TestMergedModelLocator',
           'TestChainedModelLocator']

# tested classes
from cern.cpymad.model_locator import ModelData, MergedModelLocator, ChainModelLocator

# test utilities
import unittest
import json
import os.path
import gc
import shutil
from copy import copy
from tempfile import mkdtemp
from test.libmadxtest.test_resource import create_test_file
from cern.libmadx.resource.file import FileResource


class TestMergedModelLocator(unittest.TestCase):
    """
    Test the MergedModelLocator (standard filesystem locator).
    """
    def setUp(self):
        self.base = mkdtemp()
        self.locator = MergedModelLocator(FileResource(self.base))

        # NOTE: currently, pathes are always resolved relative to the
        # derived model:
        create_test_file(self.base, ['resdata', 'a', 'a.txt'])
        create_test_file(self.base, ['resdata', 'b', 'a.txt'])
        create_test_file(self.base, ['resdata', 'c', 'a.txt'])

        create_test_file(self.base, ['repdata', 'b', 'b.txt'])
        create_test_file(self.base, ['dbdir', 'c', 'b.txt'])

        create_test_file(self.base, ['dbdir', 'c', 'c.txt'])

        abc = {
            "a": {
                # virtual models should not be listed:
                "real": False,
                "path-offsets": {
                    "repository-offset": "a",
                    "resource-offset": "a" },
                # some data that will partly be overwriten in 'b' and 'c':
                "data": {'a': 'a', 'ab': 'a', 'ac': 'a', 'abc': 'a'},
                "files": [{"path": "a.txt", "location": "RESOURCE"}],
            },
            "b": {
                "real": True,
                # subclasses 'a':
                "extends": ['a'],
                "path-offsets": {
                    "repository-offset": "b",
                    "resource-offset": "b" },
                "data": {'b': 'b', 'ab': 'b', 'bc': 'b', 'abc': 'b'},
                "files": [
                    {"path": "b.txt", "location": "REPOSITORY"},
                ],
            },
            "c": {
                "real": True,
                # subclasses 'b' -> 'a':
                "extends": ['b'],
                # different location for REPOSITORY:
                "dbdirs": [os.path.join(self.base, 'dbdir')],
                "path-offsets": {
                    "repository-offset": "c",
                    "resource-offset": "c" },
                "data": {'c': 'c', 'ac': 'c', 'bc': 'c', 'abc': 'c'},
                # missing 'location' defaults to REPOSITORY:
                "files": [{"path": "c.txt"}],
            },
        }
        de = {
            "d": abc["a"].copy(),
            "e": abc["b"].copy()
        }
        de['e']['extends'] = ['d']

        create_test_file(self.base, ['abc.cpymad.json'], json.dumps(abc))
        create_test_file(self.base, ['de.cpymad.json'], json.dumps(de))


    def tearDown(self):
        del self.locator
        gc.collect()
        shutil.rmtree(self.base)

    def test_list_models(self):
        """Verify the results of the list_models method."""
        self.assertEqual(
            set(self.locator.list_models()),
            set(('b', 'c', 'e'))) # these are the 'real' model definitions

    def test_get(self):
        """Test that ValueError is raised iff model does not exist."""
        # cannot get virtual model ATM:
        with self.assertRaises(ValueError):
            self.locator.get_model('a')
        self.locator.get_model('b')
        self.locator.get_model('c')
        with self.assertRaises(ValueError):
            self.locator.get_model('d')
        self.locator.get_model('e')
        with self.assertRaises(ValueError):
            self.locator.get_model('f')

    def test_mro(self):
        """
        Test the resolution order when using 'extends'.

        NOTE: this is not a full test of the feature, since we do not
        consider multiple extends per step here.

        """
        b = self.locator.get_model('b').model['data']
        c = self.locator.get_model('c').model['data']

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
        b = self.locator.get_model('b')
        c = self.locator.get_model('c')

        bf = b.model['files']
        cf = c.model['files']

        # the first file is defined in resdata/?/a.txt
        self.assertEqual(b.get_by_dict(bf[0]).json()['path'],
                         'resdata/b/a.txt')
        self.assertEqual(c.get_by_dict(cf[0]).json()['path'],
                         'resdata/c/a.txt')

        # second file: resdata/?/b.txt
        self.assertEqual(b.get_by_dict(bf[1]).json()['path'],
                         'repdata/b/b.txt')
        self.assertEqual(c.get_by_dict(cf[1]).json()['path'],
                         'dbdir/c/b.txt')

        # third file: repdata/?/c.txt
        self.assertEqual(c.get_by_dict(cf[2]).json()['path'],
                         'dbdir/c/c.txt')




class TestChainedModelLocator(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_(self):
        pass

