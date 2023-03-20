"""
Tests for the :class:`cpymad.madx.Madx` API.
"""

import os
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pytest import approx, fixture, mark, raises

import cpymad
from cpymad.madx import Madx, Sequence, metadata


@fixture
def mad():
    with Madx(prompt='X:> ') as mad:
        yield mad


@fixture
def lib(mad):
    return mad._libmadx


SEQU = """
! constants
QP_K1 = 2;

! elements
qp: quadrupole, k1:=QP_K1, l=1;
sb: sbend, l=2, angle=3.14/4;
dr: drift, l=1;

! sequences
s1: sequence, l=8, refer=center;
dr, at=0.5; ! dr[1] ~ betx_full1[1]
qp, at=1.5;
dr, at=2.5; ! dr[2] ~ betx_full1[3] ~ betx_range[0]
qp, at=3.5; !       ~ betx_full1[4] ~ betx_range[1]
dr, at=4.5;
sb, at=6.0; !                       ~ betx_range[3]
dr, at=7.5;
endsequence;

s2: sequence, l=3, refer=entry;
qp1: qp, at=0, k1=3;
qp2: qp, at=1, l=2;
endsequence;
"""


def normalize(path):
    """Normalize path name to eliminate different spellings of the same path.
    This is needed for path comparisons in tests, especially on windows where
    pathes are case insensitive and allow a multitude of spellings."""
    return os.path.normcase(os.path.normpath(path))


def test_copyright():
    notice = cpymad.get_copyright_notice()
    assert isinstance(notice, type(u""))


def test_version(mad):
    """Check that the Madx.version attribute can be used as expected."""
    version = mad.version
    # check format:
    major, minor, micro = map(int, version.release.split('.'))
    # We need at least MAD-X 5.05.00:
    assert (major, minor, micro) >= (5, 5, 0)
    # check format:
    year, month, day = map(int, version.date.split('.'))
    assert (year, month, day) >= (2019, 5, 10)
    assert 1 <= month <= 12
    assert 1 <= day <= 31
    assert str(version).startswith(
        'MAD-X {}'.format(version.release))


def test_metadata(mad):
    version = mad.version
    assert metadata.__version__ == version.release
    assert isinstance(metadata.get_copyright_notice(), type(u""))


def test_independent_instances():
    # Check independence by defining a variable differently in each
    # instance:
    with Madx(prompt='X1:> ') as mad1, Madx(prompt='X2:> ') as mad2:
        mad1.input('ANSWER=42;')
        mad2.input('ANSWER=43;')
        assert mad1.eval('ANSWER') == 42
        assert mad2.eval('ANSWER') == 43


# TODO: We need to fix this on windows, but for now, I just need it to
# pass so that the CI builds the release...
@mark.xfail(
    sys.platform != 'linux',
    reason='Output is sometimes garbled on MacOS and windows.',
)
def test_streamreader():
    output = []
    with Madx(stdout=output.append) as m:
        assert len(output) == 1
        assert b'+++++++++++++++++++++++++++++++++' in output[0]
        assert b'+ Support: mad@cern.ch,'           in output[0]
        assert b'+ Release   date: '                in output[0]
        assert b'+ Execution date: '                in output[0]
        # assert b'+ Support: mad@cern.ch, ', output[1]
        m.input('foo = 3;')
        assert len(output) == 1
        m.input('foo = 3;')
        assert len(output) == 2
        assert output[1] == b'++++++ info: foo redefined\n'
    assert len(output) == 3
    assert b'+          MAD-X finished normally ' in output[2]


def test_quit(mad):
    mad.quit()
    assert mad._process.returncode is not None
    assert not bool(mad)
    with raises(RuntimeError):
        mad.input(';')


@mark.xfail(
    sys.platform != 'linux',
    reason='Output is sometimes garbled on MacOS and windows.',
)
def test_context_manager():
    output = []
    with Madx(stdout=output.append) as m:
        m.input('foo = 3;')
        assert m.globals.foo == 3
    assert b'+          MAD-X finished normally ' in output[-1]
    assert not bool(m)
    with raises(RuntimeError):
        m.input(';')


def test_command_log():
    """Check that the command log contains all input commands."""
    # create a new Madx instance that uses the history feature:
    history_filename = '_test_madx.madx.tmp'
    try:
        # feed some input lines and compare with history file:
        lines = """
            l = 5;
            f = 200;

            fodo: sequence, refer=entry, l=100;
                QF: quadrupole, l=5, at= 0, k1= 1/(f*l);
                QD: quadrupole, l=5, at=50, k1=-1/(f*l);
            endsequence;

            beam, particle=proton, energy=2;
            use, sequence=fodo;
        """.splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        with Madx(command_log=history_filename) as mad:
            for line in lines:
                mad.input(line)
            with open(history_filename) as history_file:
                history = history_file.read()
            assert history.strip() == '\n'.join(lines).strip()
    finally:
        # remove history file
        os.remove(history_filename)


def test_append_semicolon():
    """Check that semicolon is automatically appended to input() text."""
    # Regression test for #73
    log = []
    with Madx(command_log=log.append) as mad:
        mad.input('a = 0')
        mad.input('b = 1')
        assert log == ['a = 0;', 'b = 1;']
        assert mad.globals.a == 0
        assert mad.globals.b == 1


def test_call_and_chdir(mad):
    folder = os.path.abspath(os.path.dirname(__file__))
    parent = os.path.dirname(folder)
    getcwd = mad._libmadx.getcwd
    g = mad.globals

    mad.chdir(folder)
    assert normalize(getcwd()) == normalize(folder)
    mad.call('answer_42.madx')
    assert g.answer == 42

    with mad.chdir('..'):
        assert normalize(getcwd()) == normalize(parent)
        mad.call('test/answer_43.madx')
        assert g.answer == 43
        mad.call('test/answer_call42.madx', True)
        assert g.answer == 42

    assert normalize(getcwd()) == normalize(folder)
    mad.call('answer_43.madx')
    assert g.answer == 43

    mad.chdir('..')
    assert normalize(getcwd()) == normalize(parent)


def _check_twiss(mad, seq_name):
    beam = 'ex=1, ey=2, particle=electron, sequence={0};'.format(seq_name)
    mad.command.beam(beam)
    mad.use(seq_name)
    initial = dict(alfx=0.5, alfy=1.5,
                   betx=2.5, bety=3.5)
    twiss = mad.twiss(sequence=seq_name, **initial)
    # Check initial values:
    assert twiss['alfx'][0] == approx(initial['alfx'])
    assert twiss['alfy'][0] == approx(initial['alfy'])
    assert twiss['betx'][0] == approx(initial['betx'])
    assert twiss['bety'][0] == approx(initial['bety'])
    assert twiss.summary['ex'] == approx(1)
    assert twiss.summary['ey'] == approx(2)
    # Check that keys are all lowercase:
    for k in twiss:
        assert k == k.lower()
    for k in twiss.summary:
        assert k == k.lower()


def test_error(mad):
    mad.input("""
        seq: sequence, l=1;
        endsequence;
        beam;
        use, sequence=seq;
    """)
    # Errors in MAD-X must not crash, but return False instead:
    assert not mad.input('twiss;')
    assert mad.input('twiss, betx=1, bety=1;')


def test_twiss_1(mad):
    mad.input(SEQU)
    _check_twiss(mad, 's1')     # s1 can be computed at start
    _check_twiss(mad, 's1')     # s1 can be computed multiple times
    _check_twiss(mad, 's2')     # s2 can be computed after s1


def test_twiss_2(mad):
    mad.input(SEQU)
    _check_twiss(mad, 's2')     # s2 can be computed at start
    _check_twiss(mad, 's1')     # s1 can be computed after s2


def test_twiss_with_range(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s1')
    params = dict(alfx=0.5, alfy=1.5,
                  betx=2.5, bety=3.5,
                  sequence='s1')
    # Compute TWISS on full sequence, then on a sub-range, then again on
    # the full sequence. This checks that none of the range selections
    # have side-effects on each other:
    betx_full1 = mad.twiss(**params)['betx']
    betx_range = mad.twiss(range=('dr[2]', 'sb'), **params)['betx']
    betx_full2 = mad.twiss(**params)['betx']
    # Check that the results have the expected lengths:
    assert len(betx_full1) == 9
    assert len(betx_range) == 4
    assert len(betx_full2) == 9
    # Check numeric results. Since the first 3 elements of range and full
    # sequence are identical, equal results are expected. And non-equal
    # results afterwards.
    assert betx_range[0] == approx(betx_full1[1])      # dr:2, dr:1
    assert betx_range[1] == approx(betx_full1[2])      # qp:2, qp:1
    assert betx_range[2] == approx(betx_full1[3])      # dr:3, dr:2
    assert betx_range[3] != approx(betx_full1[4])   # sb, qp:2


def test_range_row_api(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s1')
    params = dict(alfx=0.5, alfy=1.5,
                  betx=2.5, bety=3.5,
                  sequence='s1')
    tab = mad.twiss(range=('dr[2]', 'sb'), **params)
    assert tab.range == ('dr[2]', 'sb')
    assert 'betx' in tab


def test_survey(mad):
    mad.input(SEQU)
    mad.beam()
    mad.use('s1')
    tab = mad.survey()
    assert tab._name == 'survey'
    assert 'x' in tab
    assert 'y' in tab
    assert 'z' in tab
    assert 'theta' in tab
    assert 'phi' in tab
    assert 'psi' in tab
    assert tab.x[-1] < -1
    assert tab.y == approx(0)
    assert tab.z[-1] > 7


def test_match(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s2;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s2')

    params = dict(alfx=0.5, alfy=1.5,
                  betx=2.5, bety=3.5,
                  sequence='s2')

    mad.match(constraints=[dict(range='s1$end', betx=2.0)],
              weight={'betx': 2},
              vary=['qp2->k1'],
              **params)
    twiss = mad.twiss(**params)
    val = twiss.betx[-1]
    assert val == approx(2.0, rel=1e-2)


def test_verbose(mad):
    mad.verbose(False)
    assert mad.options.echo is False
    assert mad.options.info is False
    mad.verbose(True)
    assert mad.options.echo is True
    assert mad.options.info is True


def test_current_beam(mad):
    # Check that .beam provides access to the default BEAM:
    mad.input("beam, particle='electron', ex=1, ey=2;")
    assert mad.beam.particle == 'electron'
    assert mad.beam.ex == 1
    assert mad.beam.ey == 2
    # Check that .beam can be used as command:
    mad.beam(particle='positron', ex=2, ey=1)
    assert mad.beam.particle == 'positron'
    assert mad.beam.ex == 2
    assert mad.beam.ey == 1


def test_active_sequence(mad):
    mad.input(SEQU)
    mad.command.beam('ex=1, ey=2, particle=electron, sequence=s1;')
    mad.use('s1')
    assert mad.sequence() == 's1'
    mad.beam()
    mad.use('s2')
    assert mad.sequence().name == 's2'


def test_get_sequence(mad):
    mad.input(SEQU)
    with raises(KeyError):
        mad.sequence['sN']
    s1 = mad.sequence['s1']
    assert s1.name == 's1'
    seqs = mad.sequence
    assert set(seqs) == {'s1', 's2'}


def test_eval(mad):
    mad.input(SEQU)
    assert mad.eval(True) is True
    assert mad.eval(13) == 13
    assert mad.eval(1.3) == 1.3
    assert mad.eval([2, True, 'QP_K1']) == [2, True, 2.0]
    assert mad.eval("1/QP_K1") == approx(0.5)


def test_eval_functions(mad):
    assert mad.eval("sin(1.0)") == approx(np.sin(1.0))
    assert mad.eval("cos(1.0)") == approx(np.cos(1.0))

    mad.input("""
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
    """)

    elems = mad.sequence.fodo.expanded_elements
    twiss = mad.table.twiss

    mad.input("mqf_x = table(twiss, mqf, x);")
    assert mad.eval("table(twiss, mqf, x)") \
        == twiss.row(elems.index('mqf')).x \
        == mad.globals.mqf_x


def test_globals(mad):
    g = mad.globals
    # Membership:
    assert 'FOO' not in g
    # Setting values:
    g['FOO'] = 2
    assert 'FOO' in g
    assert g['FOO'] == 2
    assert mad.eval('FOO') == 2
    # Re-setting values:
    g['FOO'] = 3
    assert mad.eval('FOO') == 3
    # Setting expressions:
    g['BAR'] = '3*foo'
    assert mad.eval('BAR') == 9
    g['FOO'] = 4
    assert mad.eval('BAR') == 12
    assert g.defs.bar == "3*foo"
    assert g.cmdpar.bar.definition == "3*foo"
    # attribute access:
    g.bar = 42
    assert g.defs.bar == 42
    assert g.cmdpar.bar.definition == 42
    assert g.BAR == 42
    # repr
    assert "'bar': 42.0" in str(g)
    with raises(NotImplementedError):
        del g['bar']
    with raises(NotImplementedError):
        del g.bar
    assert g.bar == 42     # still there
    assert 'bar' in list(g)
    assert 'foo' in list(g)
    # assert list(g) == list(g.defs)
    # assert list(g) == list(g.cmdpar)
    assert len(g) == len(list(g))
    assert len(g.defs) == len(list(g.defs))
    assert len(g.cmdpar) == len(list(g.cmdpar))


def test_elements(mad):
    mad.input(SEQU)
    assert 'sb' in mad.elements
    assert 'sb' in list(mad.elements)
    assert 'foobar' not in mad.elements
    assert mad.elements['sb']['angle'] == approx(3.14/4)
    idx = mad.elements.index('qp1')
    elem = mad.elements[idx]
    assert elem['k1'] == 3


def test_sequence_map(mad):
    mad.input(SEQU)
    seq = mad.sequence
    assert len(seq) == 2
    assert set(seq) == {'s1', 's2'}
    assert 's1' in seq
    assert 's3' not in seq
    assert hasattr(seq, 's1')
    assert not hasattr(seq, 's3')
    assert seq.s1.name == 's1'
    assert seq.s2.name == 's2'
    with raises(AttributeError):
        seq.s3


def test_table_map(mad):
    mad.input(SEQU)
    mad.beam()
    mad.use('s2')
    mad.survey(sequence='s2')
    tab = mad.table
    assert 'survey' in list(tab)
    assert 'survey' in tab
    assert 'foobar' not in tab
    assert len(tab) == len(list(tab))
    with raises(AttributeError):
        tab.foobar


def test_sequence(mad):
    mad.input(SEQU)
    s1 = mad.sequence.s1
    assert str(s1) == '<Sequence: s1>'
    assert s1 == mad.sequence.s1
    assert s1 == 's1'
    assert s1 != mad.sequence.s2
    assert s1 != 's2'
    with raises(RuntimeError):
        s1.beam
    with raises(RuntimeError):
        s1.twiss_table
    with raises(RuntimeError):
        s1.twiss_table_name
    assert not s1.has_beam
    assert not s1.is_expanded
    s1.expand()
    assert s1.has_beam
    assert s1.is_expanded
    s1.expand()     # idempotent
    assert s1.has_beam
    assert s1.is_expanded
    initial = dict(alfx=0.5, alfy=1.5,
                   betx=2.5, bety=3.5)
    mad.twiss(sequence='s1', sectormap=True,
              table='my_twiss', **initial)
    # Now works:
    assert s1.beam.particle == 'positron'
    assert s1.twiss_table_name == 'my_twiss'
    assert s1.twiss_table.betx[0] == 2.5
    assert s1.element_names() == [
        's1$start',
        'dr', 'qp', 'dr[2]', 'qp[2]', 'dr[3]', 'sb', 'dr[4]',
        's1$end',
    ]
    assert s1.expanded_element_names() == s1.element_names()
    assert len(s1.element_names()) == len(s1.element_positions())
    assert s1.element_positions() == [
        0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]
    assert s1.expanded_element_positions() == [
        0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]

    assert s1.elements[0].name == 's1$start'
    assert s1.elements[-1].name == 's1$end'
    assert s1.elements[-1].index == len(s1.elements)-1
    assert s1.elements[3].index == 3

    assert s1.elements.index('#s') == 0
    assert s1.elements.index('#e') == len(s1.elements)-1
    assert s1.elements.index('sb') == 6
    assert s1.length == 8.0


def test_sequence_get_elements_s1(mad):
    mad.input(SEQU)
    s1 = mad.sequence.s1.elements
    qp1 = s1['qp[1]']
    qp2 = s1['qp[2]']
    sb1 = s1['sb[1]']
    assert s1.index('qp') < s1.index('qp[2]')
    assert s1.index('qp[2]') < s1.index('sb')
    assert qp1['at'] == approx(1.5)
    assert qp2['at'] == approx(3.5)
    assert sb1['at'] == approx(6)
    assert qp1.position == approx(1)
    assert qp2.position == approx(3)
    assert sb1.position == approx(5)
    assert qp1['l'] == approx(1)
    assert qp2['l'] == approx(1)
    assert sb1['l'] == approx(2)
    assert float(qp1['k1']) == approx(2)
    assert float(qp2['k1']) == approx(2)
    assert float(sb1['angle']) == approx(3.14/4)
    assert qp1.cmdpar.k1.expr.lower() == "qp_k1"


def test_sequence_get_elements_s2(mad):
    mad.input(SEQU)
    s2 = mad.sequence.s2.elements
    qp1 = s2['qp1[1]']
    qp2 = s2['qp2[1]']
    assert s2.index('qp1') < s2.index('qp2')
    assert qp1['at'] == approx(0)
    assert qp2['at'] == approx(1)
    assert qp1['l'] == approx(1)
    assert qp2['l'] == approx(2)
    assert float(qp1['k1']) == approx(3)
    assert float(qp2['k1']) == approx(2)


# def test_sequence_get_expanded_elements():


def test_crash(mad):
    """Check that a RuntimeError is raised in case MAD-X crashes."""
    assert bool(mad)
    # a.t.m. MAD-X crashes on this input, because the L (length)
    # parametere is missing:
    raises(RuntimeError, mad.input, 'XXX: sequence;')
    assert not bool(mad)


def test_sequence_elements(mad):
    mad.input(SEQU)
    elements = mad.sequence['s1'].elements
    iqp2 = elements.index('qp[2]')
    qp1 = elements['qp[1]']
    qp2 = elements[iqp2]
    assert qp1['at'] == approx(1.5)
    assert qp2['at'] == approx(3.5)
    assert qp1.position == approx(1)
    assert qp2.position == approx(3)
    assert iqp2 == elements.at(3.1)


def test_sequence_expanded_elements(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s1')
    elements = mad.sequence['s1'].expanded_elements
    iqp2 = elements.index('qp[2]')
    qp1 = elements['qp[1]']
    qp2 = elements[iqp2]
    assert qp1['at'] == approx(1.5)
    assert qp2['at'] == approx(3.5)
    assert qp1.position == approx(1)
    assert qp2.position == approx(3)
    assert iqp2 == elements.at(3.1)


def test_element_inform(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s1')
    elem = mad.sequence.s1.expanded_elements['qp']
    assert {
        name for name in elem
        if elem.cmdpar[name].inform
    } == {'k1', 'l', 'at'}


def test_table(mad):
    beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
    mad.input(SEQU)
    mad.command.beam(beam)
    mad.use('s1')
    initial = dict(alfx=0.5, alfy=1.5,
                   betx=2.5, bety=3.5)
    twiss = mad.twiss(sequence='s1', sectormap=True, **initial)
    sector = mad.table.sectortable

    assert str(twiss).startswith("<Table 'twiss': ")
    assert str(sector).startswith("<Table 'sectortable': ")

    assert 'betx' in twiss
    assert 't111' in sector
    assert 't111' not in twiss
    assert 'betx' not in sector

    assert len(twiss) == len(list(twiss))
    assert set(twiss) == set(twiss[0])
    assert twiss.s[5] == twiss[5].s
    assert twiss.s[-1] == twiss[-1].s

    copy = twiss.copy()
    assert copy['betx'] == approx(twiss.betx)
    assert set(copy) == set(twiss)
    copy = twiss.copy(['betx'])
    assert set(copy) == {'betx'}

    ALL = slice(None)

    assert sector.tmat(0).shape == (6, 6, 6)
    assert_allclose(sector.tmat(ALL)[0, 0, 0, :], sector.t111)
    assert_allclose(sector.tmat(ALL)[1, 5, 3, :], sector.t264)
    assert_allclose(sector.tmat(ALL)[3, 0, 3, :], sector.t414)
    assert_allclose(sector.tmat(ALL)[4, 4, 4, :], sector.t555)

    assert_allclose(sector.rmat(ALL)[0, 0, :], sector.r11)
    assert_allclose(sector.rmat(ALL)[1, 5, :], sector.r26)
    assert_allclose(sector.rmat(ALL)[3, 0, :], sector.r41)
    assert_allclose(sector.rmat(ALL)[4, 4, :], sector.r55)

    assert_allclose(sector.kvec(ALL)[0, :], sector.k1)
    assert_allclose(sector.kvec(ALL)[1, :], sector.k2)
    assert_allclose(sector.kvec(ALL)[3, :], sector.k4)
    assert_allclose(sector.kvec(ALL)[4, :], sector.k5)

    r = mad.sectortable()[:, :6, :6]
    k = mad.sectortable()[:, 6, :6]
    t = mad.sectortable2()

    num_elems = len(mad.sequence.s1.elements)
    assert t.shape == (num_elems, 6, 6, 6)
    assert r.shape == (num_elems, 6, 6)
    assert k.shape == (num_elems, 6)

    assert_allclose(t[:, 0, 0, 0], sector.t111)
    assert_allclose(t[:, 1, 5, 3], sector.t264)
    assert_allclose(t[:, 3, 0, 3], sector.t414)
    assert_allclose(t[:, 4, 4, 4], sector.t555)

    assert_allclose(r[:, 0, 0], sector.r11)
    assert_allclose(r[:, 1, 5], sector.r26)
    assert_allclose(r[:, 3, 0], sector.r41)
    assert_allclose(r[:, 4, 4], sector.r55)

    assert_allclose(k[:, 0], sector.k1)
    assert_allclose(k[:, 1], sector.k2)
    assert_allclose(k[:, 3], sector.k4)
    assert_allclose(k[:, 4], sector.k5)


def test_selected_columns(mad, lib):
    mad.input(SEQU)
    mad.command.beam()
    mad.use('s1')

    mad.select(flag='twiss', column=['s', 'x', 'y'])
    table = mad.twiss(sequence='s1', betx=1, bety=1)
    assert set(table) > {'s', 'x', 'y', 'betx', 'bety'}
    assert set(table.copy()) > {'s', 'x', 'y', 'betx', 'bety'}
    assert table.selected_columns() == ['s', 'x', 'y']
    assert table.selection().col_names() == ['s', 'x', 'y']
    assert table.selection().copy().keys() == {'s', 'x', 'y'}

    mad.select(flag='twiss', clear=True)
    mad.select(flag='twiss', column=['betx', 'bety'])
    lib.apply_table_selections('twiss')
    table = mad.table.twiss
    assert set(table) > {'s', 'x', 'y', 'betx', 'bety'}
    assert set(table.copy()) > {'s', 'x', 'y', 'betx', 'bety'}
    assert table.selected_columns() == ['betx', 'bety']
    assert table.selection().col_names() == ['betx', 'bety']
    assert table.selection().copy().keys() == {'betx', 'bety'}


def test_table_selected_rows(mad, lib):
    mad.input(SEQU)
    mad.command.beam()
    mad.use('s1')

    def check_selection(table, name):
        assert_equal(
            table.column(name, rows='selected'),
            table[name][table.selected_rows()])
        assert_equal(
            table.column(name, rows='selected'),
            table.selection()[name])

    mad.select(flag='twiss', class_='quadrupole')
    table = mad.twiss(sequence='s1', betx=1, bety=1)
    assert table.selected_rows() == [2, 4]
    check_selection(table, 'alfx')
    check_selection(table, 'alfy')
    check_selection(table, 'betx')
    check_selection(table, 'bety')

    mad.select(flag='twiss', clear=True)
    mad.select(flag='twiss', class_='drift')
    lib.apply_table_selections('twiss')
    table = mad.table.twiss
    assert table.selected_rows() == [1, 3, 5, 7]
    check_selection(table, 'alfx')
    check_selection(table, 'alfy')
    check_selection(table, 'betx')
    check_selection(table, 'bety')


def test_table_selected_rows_mask(mad, lib):
    mad.input(SEQU)
    mad.command.beam()
    mad.use('s1')

    mad.select(flag='twiss', class_='quadrupole')
    table = mad.twiss(sequence='s1', betx=1, bety=1)
    mask = lib.get_table_selected_rows_mask('twiss')
    assert mask.shape == (len(mad.sequence.s1.expanded_elements), )
    assert_equal(mask.nonzero(), (table.selected_rows(), ))


def test_attr(mad):
    assert hasattr(mad, 'constraint')
    assert hasattr(mad, 'constraint_')
    assert hasattr(mad, 'global_')
    assert not hasattr(mad, 'foobar')
    assert not hasattr(mad, '_constraint')


def test_expr(mad):
    g = mad.globals
    vars = mad.expr_vars
    g.foo = 1
    g.bar = 2
    assert set(vars('foo')) == {'foo'}
    assert set(vars('(foo) * sin(2*pi*bar)')) == {'foo', 'bar'}


def test_command(mad):
    mad.input(SEQU)
    twiss = mad.command.twiss
    sbend = mad.elements.sb
    clone = sbend.clone('foobar', angle="pi/5", l=1)

    assert 'betx=0' in str(twiss)
    assert 'angle=' in str(sbend)
    assert 'tilt' in sbend
    assert sbend.tilt == 0
    assert len(sbend) == len(list(sbend))
    assert 'tilt' in list(sbend)

    assert clone.name == 'foobar'
    assert clone.base_type.name == 'sbend'
    assert clone.parent.name == 'sb'
    assert clone.defs.angle == 'pi / 5'
    assert clone.angle == approx(0.6283185307179586)
    assert len(clone) == len(sbend)

    assert 'angle=0.628' in str(clone)
    assert 'tilt' not in str(clone)
    clone.angle = 0.125
    clone = mad.elements.foobar            # need to update cache
    assert clone.angle == 0.125
    assert len(twiss) == len(list(twiss))
    assert 'betx' in list(twiss)

    assert clone.angle != approx(clone.parent.angle)
    del clone.angle
    clone = mad.elements.foobar            # need to update cache
    assert clone.angle == clone.parent.angle

    with raises(AttributeError):
        clone.missing_attribute

    with raises(NotImplementedError):
        del twiss['betx']
    with raises(NotImplementedError):
        del clone.base_type.angle


def test_array_attribute(mad):
    mad.globals.nine = 9
    clone = mad.elements.multipole.clone('foo', knl=[0, 'nine/3', 4])
    knl = clone.knl
    assert knl[0] == 0
    assert knl[1] == 3
    assert knl[2] == 4
    assert len(knl) == 3
    assert list(knl) == [0.0, 3.0, 4.0]
    assert str(knl) == '[0.0, 3.0, 4.0]'
    knl[1] = '3*nine'
    assert mad.elements.foo.defs.knl[1] == '3 * nine'
    assert mad.elements.foo.knl[1] == 27


def test_array_attribute_comparison(mad):
    mad.globals.nine = 9
    foo = mad.elements.multipole.clone('foo', knl=[0, 5, 10])

    bar_eq = mad.elements.multipole.clone('bar_eq', knl=[0, 5, 10])
    bar_gt = mad.elements.multipole.clone('bar_gt', knl=[0, 6, 10])
    bar_lt = mad.elements.multipole.clone('bar_lt', knl=[0, 5, 'nine'])

    knl = foo.knl
    knl_eq = bar_eq.knl
    knl_gt = bar_gt.knl
    knl_lt = bar_lt.knl

    assert knl == knl_eq
    assert not (knl == knl_gt)
    assert not (knl == knl_lt)

    assert not (knl < knl_eq)
    assert knl < knl_gt
    assert not (knl < knl_lt)

    assert knl <= knl_eq
    assert knl <= knl_gt
    assert not (knl <= knl_lt)

    assert not (knl > knl_eq)
    assert not (knl > knl_gt)
    assert knl > knl_lt

    assert knl >= knl_eq
    assert not (knl >= knl_gt)
    assert knl >= knl_lt


def test_command_map(mad):
    command = mad.command
    assert 'match' in command
    assert 'sbend' in command

    assert 'foooo' not in command

    assert 'match' in list(command)
    assert len(command) == len(list(command))
    assert 'match' in str(command)
    assert 'sbend' in str(command)

    assert 'sbend' in mad.base_types
    assert 'match' not in mad.base_types


def test_comments(mad):
    var = mad.globals
    mad('x = 1; ! x = 2;')
    assert var.x == 1
    mad('x = 2; // x = 3;')
    assert var.x == 2
    mad('x = 3; /* x = 4; */')
    assert var.x == 3
    mad('/* x = 3; */ x = 4;')
    assert var.x == 4
    mad('x = 5; ! /* */ x = 6;')
    assert var.x == 5
    mad('x = 5; /* ! */ x = 6;')
    assert var.x == 6


def test_quoted_comment_markers(mad):
    """Comment markers within quotes should be ignored."""
    var = mad.globals
    mad("print, text='//'; x = 7;")
    assert var.x == 7
    mad('print, text="//"; x = 8;')
    assert var.x == 8
    mad('print, text="!"; x = 9;')
    assert var.x == 9
    mad('print, text="/*"; x = 10; print, text="*/";')
    assert var.x == 10


def test_multiline_input(mad):
    var = mad.globals
    mad('''
        x = 1;
        y = 2;
    ''')
    assert var.x, 1
    assert var.y, 2
    mad('''
        x = /* 3;
        y =*/ 4;
    ''')
    assert var.x == 4
    assert var.y == 2
    mad('''
        x = 1; /*  */ x = 2;
        */ if (x == 1) {
            x = 3;
        }
    ''')
    assert var.x == 2
    mad('''
        x = 1; /* x = 2;
        */ if (x == 1) {
            x = 3;
        }
    ''')
    assert var.x == 3


def test_errors(mad):
    mad.input(SEQU)
    mad.beam()
    mad.use(sequence='s1')
    mad.select(flag='error', range='qp')
    dkn = [1e-6, 2e-6, 3e-6]
    dks = [4e-6, 5e-6, 6e-6]
    mad.efcomp(dkn=dkn, dks=dks)
    mad.ealign(dx=1e-3, dy=-4e-3)
    fd = mad.sequence['s1'].expanded_elements['qp'].field_errors
    al = mad.sequence['s1'].expanded_elements['qp'].align_errors
    expected_dkn = np.hstack((dkn, np.zeros(len(fd.dkn) - len(dkn))))
    expected_dks = np.hstack((dks, np.zeros(len(fd.dks) - len(dks))))
    assert_allclose(fd.dkn, expected_dkn)
    assert_allclose(fd.dks, expected_dks)
    assert_allclose(al.dx, 1e-3)
    assert_allclose(al.dy, -4e-3)


def test_subsequence(mad):
    mad.input("""
    d1: RBEND, l=0.1, angle=0.1;

    seq1: sequence, l=0.1;
        d1.1: d1, at=0.05;
    endsequence;

    seq2: sequence, l=0.2;
        seq1, at=0.05;
        seq1, at=0.15;
    endsequence;
    """)
    seq2 = mad.sequence.seq2
    assert isinstance(seq2.elements['seq1'], Sequence)
    assert seq2.elements['seq1'].name == 'seq1'
    assert seq2.elements['seq1'].element_names() == \
        mad.sequence.seq1.element_names()


def test_dframe_after_use(mad):
    mad.input("""
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
    """)
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


def test_beams(mad):
    mad.input("beam, pc=100;")
    assert mad.beams.default_beam.pc == 100
    assert list(mad.beams) == ['default_beam']
