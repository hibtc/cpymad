"""
Regression tests.
"""

from cpymad.madx import Madx
import pytest


@pytest.fixture
def mad():
    with Madx(prompt='X:> ') as mad:
        yield mad


def test_error_table_after_clear_issue57(mad):
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

    assert len(data['name']) == 0
    assert 'name' in data
    assert 'k0l' in data


def test_no_segfault_in_makethin_issue67(mad):
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
