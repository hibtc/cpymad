"""
Unit tests for the resource components
"""

# tested classes
from cern.libmadx.resource.package import PackageResource
from cern.libmadx.resource.file import FileResource
from cern.libmadx.resource.couch import CouchResource

# test utilities
import unittest
import tempfile
import shutil
import json
import os.path
import gc
import sys
import setuptools
import contextlib
import cStringIO


def create_test_file(base, path, content=None):
    try:
        os.makedirs(os.path.join(base, *path[:-1]))
    except OSError:
        # directory already exists. the exist_ok parameter exists not until
        # python3.2
        pass
    with open(os.path.join(base, *path), 'w') as f:
        if content is None:
            json.dump({'path': os.path.join(*path)}, f)
        else:
            f.write(content)

def set_(iterable):
    return set((s for s in iterable if not s.endswith('.pyc')))


@contextlib.contextmanager
def captured_output(stream_name):
    orig_stdout = getattr(sys, stream_name)
    setattr(sys, stream_name, cStringIO.StringIO())
    try:
        yield getattr(sys, stream_name)
    finally:
        setattr(sys, stream_name, orig_stdout)


# common test code

class Common(object):
    def setUp(self):
        self.mod = 'dummy_mod_124'
        self.base = tempfile.mkdtemp()
        self.path = os.path.join(self.base, self.mod)
        create_test_file(self.path, ['__init__.py'], '')
        create_test_file(self.path, ['a.json'])
        create_test_file(self.path, ['subdir', 'b.json'])

    def tearDown(self):
        gc.collect()
        shutil.rmtree(self.base)

    def test_open(self):
        with self.res.open('a.json') as f:
            self.assertEqual(
                    json.load(f)['path'],
                    'a.json')
        with self.res.get('subdir/b.json').open() as f:
            self.assertEqual(
                    json.load(f)['path'],
                    os.path.join('subdir', 'b.json'))

    def test_list(self):
        self.assertEqual(
                set_(self.res.listdir()),
                set(('__init__.py', 'a.json', 'subdir')))
        self.assertEqual(
                set(self.res.listdir('subdir')),
                set(('b.json',)))
        self.assertEqual(
                set(self.res.get('subdir').listdir()),
                set(('b.json',)))

    def test_provider(self):
        self.assertEqual(
                set_(self.res.get('subdir').provider().listdir()),
                set(('__init__.py', 'a.json', 'subdir')))
        self.assertEqual(
                set(self.res.get(['subdir', 'b.json']).provider().listdir()),
                set(('b.json',)))

    def test_filter(self):
        self.assertEqual(
                set(self.res.listdir_filter(ext='.txt')),
                set())
        self.assertEqual(
                set(self.res.listdir_filter(ext='.json')),
                set(('a.json',)))

    def test_load(self):
        self.assertEqual(
                json.loads(self.res.load('a.json'))['path'],
                'a.json')
        self.assertEqual(
                json.loads(self.res.get('subdir').load('b.json'))['path'],
                os.path.join('subdir', 'b.json'))

    def test_json(self):
        self.assertEqual(
                self.res.json('a.json')['path'],
                'a.json')
        self.assertEqual(
                self.res.get(['subdir', 'b.json']).json()['path'],
                os.path.join('subdir', 'b.json'))


# test cases
class TestPackageResource(Common, unittest.TestCase):
    def setUp(self):
        super(TestPackageResource, self).setUp()
        sys.path.append(self.base)
        self.res = PackageResource(self.mod)

    def tearDown(self):
        del self.res
        del sys.modules[self.mod]
        sys.path.remove(self.base)
        super(TestPackageResource, self).tearDown()

    def test_filename(self):
        with self.res.filename('a.json') as filename:
            with open(filename) as f:
                self.assertEqual(
                        json.load(f)['path'],
                        'a.json')
        with self.res.get(['subdir', 'b.json']).filename() as filename:
            with open(filename) as f:
                self.assertEqual(
                        json.load(f)['path'],
                        'subdir/b.json')

class TestEggResource(Common, unittest.TestCase):
    def setUp(self):
        super(TestEggResource, self).setUp()
        cwd = os.getcwd()
        os.chdir(self.base)
        with captured_output('stdout'), captured_output('stderr'):
            setuptools.setup(
                    name=self.mod,
                    packages=[self.mod],
                    script_args=['bdist_egg', '--quiet'],
                    package_data={self.mod:[
                        'a.json',
                        os.path.join('subdir', 'b.json')]}
                    )
        os.chdir(cwd)
        self.eggs = os.listdir(os.path.join(self.base, 'dist'))
        for egg in self.eggs:
            sys.path.append(os.path.join(self.base, 'dist', egg))
        self.res = PackageResource(self.mod)

    def tearDown(self):
        del self.res
        del sys.modules[self.mod]
        for egg in self.eggs:
            sys.path.remove(os.path.join(self.base, 'dist', egg))
        super(TestEggResource, self).tearDown()

    def test_filename(self):
        with self.res.filename('a.json') as filename:
            with open(filename) as f:
                self.assertEqual(
                        json.load(f)['path'],
                        'a.json')
        self.assertFalse(os.path.exists(filename))
        with self.res.get(['subdir', 'b.json']).filename() as filename:
            with open(filename) as f:
                self.assertEqual(
                        json.load(f)['path'],
                        'subdir/b.json')
        self.assertFalse(os.path.exists(filename))

class TestFileResource(Common, unittest.TestCase):
    def setUp(self):
        super(TestFileResource, self).setUp()
        self.res = FileResource(self.path)

    def tearDown(self):
        del self.res
        super(TestFileResource, self).tearDown()

    def test_filename(self):
        with self.res.filename('a.json') as filename:
            self.assertEqual(
                    filename,
                    os.path.join(self.path, 'a.json'))
        with self.res.get(['subdir', 'b.json']).filename() as filename:
            self.assertEqual(
                    filename,
                    os.path.join(self.path, 'subdir', 'b.json'))


class TestCouchResource(unittest.TestCase):
    """TODO."""
    pass

