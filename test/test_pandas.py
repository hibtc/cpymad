"""
Tests that involve pandas data frames.

These tests are not included within the main tests, so that they can be
executed selectively.
"""

from cpymad.madx import Madx


SEQU = """
mqf.k1 =  0.3037241107;
mqd.k1 = -0.3037241107;

fodo: sequence, l=10, refer=entry;
mqf: quadrupole, at=0, l=1, k1:=mqf.k1;
dff: drift,      at=1, l=4;
mqd: quadrupole, at=5, l=1, k1:=mqd.k1;
dfd: drift,      at=6, l=4;
endsequence;

beam;
use, sequence=fodo;
twiss, sequence=fodo, x=0.1;
"""


def test_dframe_after_use():
    with Madx(prompt='X:> ') as mad:
        mad.input(SEQU)
        index = ['#s', 'mqf', 'dff', 'mqd', 'dfd', '#e']
        names = ['fodo$start', 'mqf', 'dff', 'mqd', 'dfd', 'fodo$end']

        twiss = mad.table.twiss
        assert index == twiss.row_names()
        assert index == twiss.dframe().index.tolist()
        assert names == twiss.dframe(index='name').index.tolist()

        mad.use(sequence='fodo')

        twiss = mad.table.twiss

        # Should still work:
        assert names == twiss.dframe(index='name').index.tolist()

        # The following assert demonstrates the current behaviour and is
        # meant to detect if the MAD-X implementation changes. It may lead
        # to crashes or change in the future. In that case, please remove
        # this line. It does not represent desired behaviour!
        assert mad.table.twiss.row_names() == \
            ['#s', '#e', 'dfd', 'mqd', 'dff', 'mqf']
