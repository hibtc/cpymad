import sys
import unittest

from cpymad.madx import Madx, CommandLog


class RegressionTests(unittest.TestCase):

    def setUp(self):
        self.madx = Madx(command_log=CommandLog(sys.stdout, 'X:> '))

    def tearDown(self):
        self.madx.quit()
        del self.madx

    def test_error_table_after_clear_issue57(self):
        """
        Test that ``Madx.table.error`` works as expected.
        """
        # See: https://github.com/hibtc/cpymad/issues/57
        madx = self.madx
        madx.verbose(False)
        madx.input("""
        fodo: sequence, l=10, refer=entry;
        endsequence;
        beam;
        use, sequence=fodo;
        select, flag=error, clear;
        etable, table=error;
        """)
        # The following line would previously cause a:
        #   KeyError: "Unknown table column: 'k0l'"
        data = madx.table.error.copy()

        self.assertEqual(len(data['name']), 0)
        self.assertIn('name', data)
        self.assertIn('k0l', data)


if __name__ == '__main__':
    unittest.main()
