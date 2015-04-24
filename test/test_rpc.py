# standard library
import os
import subprocess
import unittest

# utilities
import _compat

# tested modules
from cpymad import _rpc


class TestRPC(unittest.TestCase, _compat.TestCase):

    # This test is currently only useful for windows. On linux open file
    # descriptors don't prevent the file from being deleted.
    # TODO: change this test such that it checks "directly" whether the file
    # handle is still open.
    def test_no_leaking_file_handles(self):
        test_filename = 'foobar'
        f = open(test_filename, 'w')
        print(f.fileno())
        svc, proc = _rpc.LibMadxClient.spawn_subprocess()
        f.close()
        try:
            os.remove(test_filename)
            file_can_be_deleted = True
        except OSError:
            file_can_be_deleted = False
        svc.close()
        proc.wait()
        if not file_can_be_deleted:
            os.remove(test_filename)
        self.assertTrue(file_can_be_deleted)

    # TODO: add tests to check that other resources get closed correctly.

if __name__ == '__main__':
    unittest.main()