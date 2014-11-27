# encoding: utf-8
"""
Tests for the classes defined in cern.cpymad.model_locator.
"""
__all__ = ['TestMergedModelLocator',
           'TestChainedModelLocator',
           ]

# tested classes
from cern.cpymad.model_locator import ModelData, MergedModelLocator, ChainModelLocator, C3_mro

# test utilities
import unittest
import json
import os.path
import gc
import shutil
from copy import copy
from tempfile import mkdtemp
from io import open
from cern.resource.file import FileResource


def create_test_file(base, path, content=None):
    """
    Create a file with defined content under base/path.
    """
    try:
        os.makedirs(os.path.join(base, *path[:-1]))
    except OSError:
        # directory already exists. the exist_ok parameter exists not until
        # python3.2
        pass
    with open(os.path.join(base, *path), 'wt', encoding='utf-8') as f:
        if content is None:
            # With json.dump is not compatible in python2 and python3
            f.write(u'{"path": "%s", "unicode": "%s"}' % (
                os.path.join(*path),    # this content is predictable
                u"äæo≤»で"))            # some unicode test data
        else:
            f.write(content)



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

        self.unicode_data = u"€®æة《±∓"
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
                # some unicode test data:
                "unicode": self.unicode_data
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

        create_test_file(self.base,
                         ['abc.cpymad.yml'],
                         json.dumps(abc, ensure_ascii=False))
        create_test_file(self.base,
                         ['de.cpymad.yml'],
                         json.dumps(de, ensure_ascii=False))


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
        self.assertRaises(ValueError, self.locator.get_model, 'a')
        self.locator.get_model('b')
        self.locator.get_model('c')
        self.assertRaises(ValueError, self.locator.get_model, 'd')
        self.locator.get_model('e')
        self.assertRaises(ValueError, self.locator.get_model, 'f')

    def test_encoding(self):
        """Test that the model is loaded with the correct encoding."""
        b = self.locator.get_model('b').model
        self.assertEqual(b['unicode'],
                         self.unicode_data)

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
        self.assertEqual(b.get_by_dict(bf[0]).yaml()['path'],
                         'resdata/b/a.txt')
        self.assertEqual(c.get_by_dict(cf[0]).yaml()['path'],
                         'resdata/c/a.txt')

        # second file: resdata/?/b.txt
        self.assertEqual(b.get_by_dict(bf[1]).yaml()['path'],
                         'repdata/b/b.txt')
        self.assertEqual(c.get_by_dict(cf[1]).yaml()['path'],
                         'dbdir/c/b.txt')

        # third file: repdata/?/c.txt
        self.assertEqual(c.get_by_dict(cf[2]).yaml()['path'],
                         'dbdir/c/c.txt')


if __name__ == '__main__':
    unittest.main()
