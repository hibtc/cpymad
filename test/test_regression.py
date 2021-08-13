import unittest

from common import with_madx


class RegressionTests(unittest.TestCase):

    @with_madx()
    def test_error_table_after_clear_issue57(self, mad):
        """
        Test that ``Madx.table.error`` works as expected.
        """
        # See: https://github.com/hibtc/cpymad/issues/57
        mad.verbose(False)
        mad.input("""
        fodo: sequence, l=10, refer=entry;
        endsequence;
        beam;
        use, sequence=fodo;
        select, flag=error, clear;
        etable, table=error;
        """)
        # The following line would previously cause a:
        #   KeyError: "Unknown table column: 'k0l'"
        data = mad.table.error.copy()

        self.assertEqual(len(data['name']), 0)
        self.assertIn('name', data)
        self.assertIn('k0l', data)

    @with_madx()
    def test_no_segfault_in_makethin_issue67(self, mad):
        # See: https://github.com/hibtc/cpymad/issues/67
        mad.input("""
            seq: sequence, l=2, refer=center;
            q1: quadrupole, l=1, at=1;
            endsequence;

            beam;
            use, sequence=seq;

            select, flag=MAKETHIN, class=quadrupole;
            makethin, sequence=seq;
        """)


if __name__ == '__main__':
    unittest.main()
