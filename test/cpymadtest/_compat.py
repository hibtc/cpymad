
class TestCase(object):

    """
    Compatibility layer for unittest.TestCase
    """

    def assertItemsEqual(self, first, second):
        """Method missing in python2.6 and renamed in python3."""
        self.assertEqual(sorted(first), sorted(second))

    def assertLess(self, first, second):
        """Method missing in python2.6."""
        self.assertTrue(first < second)

