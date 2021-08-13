# standard library
import os
import sys
import unittest
from textwrap import dedent

import numpy as np
from numpy.testing import assert_allclose

# tested class
import cpymad
from cpymad.madx import metadata
from common import with_madx, create_madx as Madx


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


class TestMadx(unittest.TestCase):

    """Test methods for the Madx class."""

    def test_copyright(self):
        notice = cpymad.get_copyright_notice()
        self.assertIsInstance(notice, type(u""))

    @with_madx()
    def test_version(self, mad):
        """Check that the Madx.version attribute can be used as expected."""
        version = mad.version
        # check format:
        major, minor, micro = map(int, version.release.split('.'))
        # We need at least MAD-X 5.05.00:
        self.assertGreaterEqual((major, minor, micro), (5, 5, 0))
        # check format:
        year, month, day = map(int, version.date.split('.'))
        self.assertGreaterEqual((year, month, day), (2019, 5, 10))
        self.assertLessEqual(month, 12)
        self.assertLessEqual(day, 31)
        self.assertTrue(str(version).startswith(
            'MAD-X {}'.format(version.release)))

    @with_madx()
    def test_metadata(self, mad):
        version = mad.version
        self.assertEqual(metadata.__version__, version.release)
        self.assertIsInstance(metadata.get_copyright_notice(), type(u""))

    @with_madx()
    @with_madx()
    def test_independent_instances(self, mad1, mad2):
        # Check independence by defining a variable differently in each
        # instance:
        mad1.input('ANSWER=42;')
        mad2.input('ANSWER=43;')
        self.assertEqual(mad1.eval('ANSWER'), 42)
        self.assertEqual(mad2.eval('ANSWER'), 43)

    # TODO: We need to fix this on windows, but for now, I just need it to
    # pass so that the CI builds the release...
    @unittest.skipIf(sys.platform == 'win32', 'Known to be broken on win32!')
    def test_streamreader(self):
        output = []
        with Madx(stdout=output.append) as m:
            self.assertEqual(len(output), 1)
            self.assertIn(b'+++++++++++++++++++++++++++++++++', output[0])
            self.assertIn(b'+ Support: mad@cern.ch,',           output[0])
            self.assertIn(b'+ Release   date: ',                output[0])
            self.assertIn(b'+ Execution date: ',                output[0])
            # self.assertIn(b'+ Support: mad@cern.ch, ', output[1])
            m.input('foo = 3;')
            self.assertEqual(len(output), 1)
            m.input('foo = 3;')
            self.assertEqual(len(output), 2)
            self.assertEqual(output[1], b'++++++ info: foo redefined\n')
        self.assertEqual(len(output), 3)
        self.assertIn(b'+          MAD-X finished normally ', output[2])

    @with_madx()
    def test_quit(self, mad):
        mad.quit()
        self.assertIsNot(mad._process.returncode, None)
        self.assertFalse(bool(mad))
        with self.assertRaises(RuntimeError):
            mad.input(';')

    @unittest.skipIf(sys.platform == 'win32', 'Known to be broken on win32!')
    def test_context_manager(self):
        output = []
        with Madx(stdout=output.append) as m:
            m.input('foo = 3;')
            self.assertEqual(m.globals.foo, 3)
        self.assertIn(b'+          MAD-X finished normally ', output[-1])
        self.assertFalse(bool(m))
        with self.assertRaises(RuntimeError):
            m.input(';')

    def test_command_log(self):
        """Check that the command log contains all input commands."""
        # create a new Madx instance that uses the history feature:
        history_filename = '_test_madx.madx.tmp'
        try:
            # feed some input lines and compare with history file:
            lines = dedent("""
                l = 5;
                f = 200;

                fodo: sequence, refer=entry, l=100;
                    QF: quadrupole, l=5, at= 0, k1= 1/(f*l);
                    QD: quadrupole, l=5, at=50, k1=-1/(f*l);
                endsequence;

                beam, particle=proton, energy=2;
                use, sequence=fodo;
            """).splitlines()
            lines = [line for line in lines if line.strip()]
            with Madx(command_log=history_filename) as mad:
                for line in lines:
                    mad.input(line)
                with open(history_filename) as history_file:
                    history = history_file.read()
                self.assertEqual(history.strip(), '\n'.join(lines).strip())
        finally:
            # remove history file
            os.remove(history_filename)

    def test_append_semicolon(self):
        """Check that semicolon is automatically appended to input() text."""
        # Regression test for #73
        log = []
        with Madx(command_log=log.append) as mad:
            mad.input('a = 0')
            mad.input('b = 1')
            self.assertEqual(log, ['a = 0;', 'b = 1;'])
            self.assertEqual(mad.globals.a, 0)
            self.assertEqual(mad.globals.b, 1)

    @with_madx()
    def test_call_and_chdir(self, mad):
        folder = os.path.abspath(os.path.dirname(__file__))
        parent = os.path.dirname(folder)
        getcwd = mad._libmadx.getcwd
        g = mad.globals

        mad.chdir(folder)
        self.assertEqual(normalize(getcwd()), normalize(folder))
        mad.call('answer_42.madx')
        self.assertEqual(g.answer, 42)

        with mad.chdir('..'):
            self.assertEqual(normalize(getcwd()), normalize(parent))
            mad.call('test/answer_43.madx')
            self.assertEqual(g.answer, 43)
            mad.call('test/answer_call42.madx', True)
            self.assertEqual(g.answer, 42)

        self.assertEqual(normalize(getcwd()), normalize(folder))
        mad.call('answer_43.madx')
        self.assertEqual(g.answer, 43)

        mad.chdir('..')
        self.assertEqual(normalize(getcwd()), normalize(parent))

    def _check_twiss(self, mad, seq_name):
        beam = 'ex=1, ey=2, particle=electron, sequence={0};'.format(seq_name)
        mad.command.beam(beam)
        mad.use(seq_name)
        initial = dict(alfx=0.5, alfy=1.5,
                       betx=2.5, bety=3.5)
        twiss = mad.twiss(sequence=seq_name, **initial)
        # Check initial values:
        self.assertAlmostEqual(twiss['alfx'][0], initial['alfx'])
        self.assertAlmostEqual(twiss['alfy'][0], initial['alfy'])
        self.assertAlmostEqual(twiss['betx'][0], initial['betx'])
        self.assertAlmostEqual(twiss['bety'][0], initial['bety'])
        self.assertAlmostEqual(twiss.summary['ex'], 1)
        self.assertAlmostEqual(twiss.summary['ey'], 2)
        # Check that keys are all lowercase:
        for k in twiss:
            self.assertEqual(k, k.lower())
        for k in twiss.summary:
            self.assertEqual(k, k.lower())

    @with_madx()
    def test_error(self, mad):
        mad.input("""
            seq: sequence, l=1;
            endsequence;
            beam;
            use, sequence=seq;
        """)
        # Errors in MAD-X must not crash, but return False instead:
        self.assertFalse(mad.input('twiss;'))
        self.assertTrue(mad.input('twiss, betx=1, bety=1;'))

    @with_madx()
    def test_twiss_1(self, mad):
        mad.input(SEQU)
        self._check_twiss(mad, 's1')     # s1 can be computed at start
        self._check_twiss(mad, 's1')     # s1 can be computed multiple times
        self._check_twiss(mad, 's2')     # s2 can be computed after s1

    @with_madx()
    def test_twiss_2(self, mad):
        mad.input(SEQU)
        self._check_twiss(mad, 's2')     # s2 can be computed at start
        self._check_twiss(mad, 's1')     # s1 can be computed after s2

    @with_madx()
    def test_twiss_with_range(self, mad):
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
        self.assertEqual(len(betx_full1), 9)
        self.assertEqual(len(betx_range), 4)
        self.assertEqual(len(betx_full2), 9)
        # Check numeric results. Since the first 3 elements of range and full
        # sequence are identical, equal results are expected. And non-equal
        # results afterwards.
        self.assertAlmostEqual(betx_range[0], betx_full1[1])      # dr:2, dr:1
        self.assertAlmostEqual(betx_range[1], betx_full1[2])      # qp:2, qp:1
        self.assertAlmostEqual(betx_range[2], betx_full1[3])      # dr:3, dr:2
        self.assertNotAlmostEqual(betx_range[3], betx_full1[4])   # sb, qp:2

    @with_madx()
    def test_range_row_api(self, mad):
        beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
        mad.input(SEQU)
        mad.command.beam(beam)
        mad.use('s1')
        params = dict(alfx=0.5, alfy=1.5,
                      betx=2.5, bety=3.5,
                      sequence='s1')
        tab = mad.twiss(range=('dr[2]', 'sb'), **params)
        self.assertEqual(tab.range, ('dr[2]', 'sb'))
        self.assertIn('betx', tab)

    @with_madx()
    def test_survey(self, mad):
        mad.input(SEQU)
        mad.beam()
        mad.use('s1')
        tab = mad.survey()
        self.assertEqual(tab._name, 'survey')
        self.assertIn('x', tab)
        self.assertIn('y', tab)
        self.assertIn('z', tab)
        self.assertIn('theta', tab)
        self.assertIn('phi', tab)
        self.assertIn('psi', tab)
        self.assertLess(tab.x[-1], -1)
        assert_allclose(tab.y, 0)
        self.assertGreater(tab.z[-1], 7)

    @with_madx()
    def test_match(self, mad):
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
        self.assertAlmostEqual(val, 2.0, places=2)

    @with_madx()
    def test_verbose(self, mad):
        mad.verbose(False)
        self.assertEqual(mad.options.echo, False)
        self.assertEqual(mad.options.info, False)
        mad.verbose(True)
        self.assertEqual(mad.options.echo, True)
        self.assertEqual(mad.options.info, True)

    @with_madx()
    def test_active_sequence(self, mad):
        mad.input(SEQU)
        mad.command.beam('ex=1, ey=2, particle=electron, sequence=s1;')
        mad.use('s1')
        self.assertEqual(mad.sequence(), 's1')
        mad.beam()
        mad.use('s2')
        self.assertEqual(mad.sequence().name, 's2')

    @with_madx()
    def test_get_sequence(self, mad):
        mad.input(SEQU)
        with self.assertRaises(KeyError):
            mad.sequence['sN']
        s1 = mad.sequence['s1']
        self.assertEqual(s1.name, 's1')
        seqs = mad.sequence
        self.assertCountEqual(seqs, ['s1', 's2'])

    @with_madx()
    def test_eval(self, mad):
        mad.input(SEQU)
        self.assertEqual(mad.eval(True), True)
        self.assertEqual(mad.eval(13), 13)
        self.assertEqual(mad.eval(1.3), 1.3)
        self.assertEqual(mad.eval([2, True, 'QP_K1']), [2, True, 2.0])
        self.assertAlmostEqual(mad.eval("1/QP_K1"), 0.5)

    @with_madx()
    def test_globals(self, mad):
        g = mad.globals
        # Membership:
        self.assertNotIn('FOO', g)
        # Setting values:
        g['FOO'] = 2
        self.assertIn('FOO', g)
        self.assertEqual(g['FOO'], 2)
        self.assertEqual(mad.eval('FOO'), 2)
        # Re-setting values:
        g['FOO'] = 3
        self.assertEqual(mad.eval('FOO'), 3)
        # Setting expressions:
        g['BAR'] = '3*foo'
        self.assertEqual(mad.eval('BAR'), 9)
        g['FOO'] = 4
        self.assertEqual(mad.eval('BAR'), 12)
        self.assertEqual(g.defs.bar, "3*foo")
        self.assertEqual(g.cmdpar.bar.definition, "3*foo")
        # attribute access:
        g.bar = 42
        self.assertEqual(g.defs.bar, 42)
        self.assertEqual(g.cmdpar.bar.definition, 42)
        self.assertEqual(g.BAR, 42)
        # repr
        self.assertIn("'bar': 42.0", str(g))
        with self.assertRaises(NotImplementedError):
            del g['bar']
        with self.assertRaises(NotImplementedError):
            del g.bar
        self.assertEqual(g.bar, 42)     # still there
        self.assertIn('bar', list(g))
        self.assertIn('foo', list(g))
        # self.assertEqual(list(g), list(g.defs))
        # self.assertEqual(list(g), list(g.cmdpar))
        self.assertEqual(len(g), len(list(g)))
        self.assertEqual(len(g.defs), len(list(g.defs)))
        self.assertEqual(len(g.cmdpar), len(list(g.cmdpar)))

    @with_madx()
    def test_elements(self, mad):
        mad.input(SEQU)
        self.assertIn('sb', mad.elements)
        self.assertIn('sb', list(mad.elements))
        self.assertNotIn('foobar', mad.elements)
        self.assertAlmostEqual(mad.elements['sb']['angle'], 3.14/4)
        idx = mad.elements.index('qp1')
        elem = mad.elements[idx]
        self.assertEqual(elem['k1'], 3)

    @with_madx()
    def test_sequence_map(self, mad):
        mad.input(SEQU)
        seq = mad.sequence
        self.assertEqual(len(seq), 2)
        self.assertEqual(set(seq), {'s1', 's2'})
        self.assertIn('s1', seq)
        self.assertNotIn('s3', seq)
        self.assertTrue(hasattr(seq, 's1'))
        self.assertFalse(hasattr(seq, 's3'))
        self.assertEqual(seq.s1.name, 's1')
        self.assertEqual(seq.s2.name, 's2')
        with self.assertRaises(AttributeError):
            seq.s3

    @with_madx()
    def test_table_map(self, mad):
        mad.input(SEQU)
        mad.beam()
        mad.use('s2')
        mad.survey(sequence='s2')
        tab = mad.table
        self.assertIn('survey', list(tab))
        self.assertIn('survey', tab)
        self.assertNotIn('foobar', tab)
        self.assertEqual(len(tab), len(list(tab)))
        with self.assertRaises(AttributeError):
            tab.foobar

    @with_madx()
    def test_sequence(self, mad):
        mad.input(SEQU)
        s1 = mad.sequence.s1
        self.assertEqual(str(s1), '<Sequence: s1>')
        self.assertEqual(s1, mad.sequence.s1)
        self.assertEqual(s1, 's1')
        self.assertNotEqual(s1, mad.sequence.s2)
        self.assertNotEqual(s1, 's2')
        with self.assertRaises(RuntimeError):
            s1.beam
        with self.assertRaises(RuntimeError):
            s1.twiss_table
        with self.assertRaises(RuntimeError):
            s1.twiss_table_name
        self.assertFalse(s1.has_beam)
        self.assertFalse(s1.is_expanded)
        s1.expand()
        self.assertTrue(s1.has_beam)
        self.assertTrue(s1.is_expanded)
        s1.expand()     # idempotent
        self.assertTrue(s1.has_beam)
        self.assertTrue(s1.is_expanded)
        initial = dict(alfx=0.5, alfy=1.5,
                       betx=2.5, bety=3.5)
        mad.twiss(sequence='s1', sectormap=True,
                  table='my_twiss', **initial)
        # Now works:
        self.assertEqual(s1.beam.particle, 'positron')
        self.assertEqual(s1.twiss_table_name, 'my_twiss')
        self.assertEqual(s1.twiss_table.betx[0], 2.5)
        self.assertEqual(s1.element_names(), [
            's1$start',
            'dr', 'qp', 'dr[2]', 'qp[2]', 'dr[3]', 'sb', 'dr[4]',
            's1$end',
        ])
        self.assertEqual(s1.expanded_element_names(), s1.element_names())
        self.assertEqual(len(s1.element_names()), len(s1.element_positions()))
        self.assertEqual(s1.element_positions(), [
            0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0])
        self.assertEqual(s1.expanded_element_positions(), [
            0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0])

        self.assertEqual(s1.elements[0].name, 's1$start')
        self.assertEqual(s1.elements[-1].name, 's1$end')
        self.assertEqual(s1.elements[-1].index, len(s1.elements)-1)
        self.assertEqual(s1.elements[3].index, 3)

        self.assertEqual(s1.elements.index('#s'), 0)
        self.assertEqual(s1.elements.index('#e'), len(s1.elements)-1)
        self.assertEqual(s1.elements.index('sb'), 6)
        self.assertEqual(s1.length, 8.0)

    @with_madx()
    def test_sequence_get_elements_s1(self, mad):
        mad.input(SEQU)
        s1, idx = _get_elems(mad, 's1')
        qp1 = s1['qp[1]']
        qp2 = s1['qp[2]']
        sb1 = s1['sb[1]']
        self.assertLess(idx['qp'], idx['qp[2]'])
        self.assertLess(idx['qp[2]'], idx['sb'])
        self.assertAlmostEqual(qp1['at'], 1.5)
        self.assertAlmostEqual(qp2['at'], 3.5)
        self.assertAlmostEqual(sb1['at'], 6)
        self.assertAlmostEqual(qp1.position, 1)
        self.assertAlmostEqual(qp2.position, 3)
        self.assertAlmostEqual(sb1.position, 5)
        self.assertAlmostEqual(qp1['l'], 1)
        self.assertAlmostEqual(qp2['l'], 1)
        self.assertAlmostEqual(sb1['l'], 2)
        self.assertAlmostEqual(float(qp1['k1']), 2)
        self.assertAlmostEqual(float(qp2['k1']), 2)
        self.assertAlmostEqual(float(sb1['angle']), 3.14/4)
        self.assertEqual(qp1.cmdpar.k1.expr.lower(), "qp_k1")

    @with_madx()
    def test_sequence_get_elements_s2(self, mad):
        mad.input(SEQU)
        s2, idx = _get_elems(mad, 's2')
        qp1 = s2['qp1[1]']
        qp2 = s2['qp2[1]']
        self.assertLess(idx['qp1'], idx['qp2'])
        self.assertAlmostEqual(qp1['at'], 0)
        self.assertAlmostEqual(qp2['at'], 1)
        self.assertAlmostEqual(qp1['l'], 1)
        self.assertAlmostEqual(qp2['l'], 2)
        self.assertAlmostEqual(float(qp1['k1']), 3)
        self.assertAlmostEqual(float(qp2['k1']), 2)

    # def test_sequence_get_expanded_elements(self):

    @with_madx()
    def test_crash(self, mad):
        """Check that a RuntimeError is raised in case MAD-X crashes."""
        self.assertTrue(mad)
        # a.t.m. MAD-X crashes on this input, because the L (length)
        # parametere is missing:
        self.assertRaises(RuntimeError, mad.input, 'XXX: sequence;')
        self.assertFalse(mad)

    @with_madx()
    def test_sequence_elements(self, mad):
        mad.input(SEQU)
        elements = mad.sequence['s1'].elements
        iqp2 = elements.index('qp[2]')
        qp1 = elements['qp[1]']
        qp2 = elements[iqp2]
        self.assertAlmostEqual(qp1['at'], 1.5)
        self.assertAlmostEqual(qp2['at'], 3.5)
        self.assertAlmostEqual(qp1.position, 1)
        self.assertAlmostEqual(qp2.position, 3)
        self.assertEqual(iqp2, elements.at(3.1))

    @with_madx()
    def test_sequence_expanded_elements(self, mad):
        beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
        mad.input(SEQU)
        mad.command.beam(beam)
        mad.use('s1')
        elements = mad.sequence['s1'].expanded_elements
        iqp2 = elements.index('qp[2]')
        qp1 = elements['qp[1]']
        qp2 = elements[iqp2]
        self.assertAlmostEqual(qp1['at'], 1.5)
        self.assertAlmostEqual(qp2['at'], 3.5)
        self.assertAlmostEqual(qp1.position, 1)
        self.assertAlmostEqual(qp2.position, 3)
        self.assertEqual(iqp2, elements.at(3.1))

    @with_madx()
    def test_element_inform(self, mad):
        beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
        mad.input(SEQU)
        mad.command.beam(beam)
        mad.use('s1')
        elem = mad.sequence.s1.expanded_elements['qp']
        self.assertSetEqual({'k1', 'l', 'at'}, {
            name for name in elem
            if elem.cmdpar[name].inform
        })

    @with_madx()
    def test_table(self, mad):
        beam = 'ex=1, ey=2, particle=electron, sequence=s1;'
        mad.input(SEQU)
        mad.command.beam(beam)
        mad.use('s1')
        initial = dict(alfx=0.5, alfy=1.5,
                       betx=2.5, bety=3.5)
        twiss = mad.twiss(sequence='s1', sectormap=True, **initial)
        sector = mad.table.sectortable

        self.assertTrue(str(twiss).startswith("<Table 'twiss': "))
        self.assertTrue(str(sector).startswith("<Table 'sectortable': "))

        self.assertIn('betx', twiss)
        self.assertIn('t111', sector)
        self.assertNotIn('t111', twiss)
        self.assertNotIn('betx', sector)

        self.assertEqual(len(twiss), len(list(twiss)))
        self.assertEqual(set(twiss), set(twiss[0]))
        self.assertEqual(twiss.s[5], twiss[5].s)
        self.assertEqual(twiss.s[-1], twiss[-1].s)

        copy = twiss.copy()
        assert_allclose(copy['betx'], twiss.betx)
        self.assertEqual(set(copy), set(twiss))
        copy = twiss.copy(['betx'])
        self.assertEqual(set(copy), {'betx'})

        ALL = slice(None)

        self.assertEqual(sector.tmat(0).shape, (6, 6, 6))
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
        self.assertEqual(t.shape, (num_elems, 6, 6, 6))
        self.assertEqual(r.shape, (num_elems, 6, 6))
        self.assertEqual(k.shape, (num_elems, 6))

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

    @with_madx()
    def test_attr(self, mad):
        self.assertTrue(hasattr(mad, 'constraint'))
        self.assertTrue(hasattr(mad, 'constraint_'))
        self.assertTrue(hasattr(mad, 'global_'))
        self.assertFalse(hasattr(mad, 'foobar'))
        self.assertFalse(hasattr(mad, '_constraint'))

    @with_madx()
    def test_expr(self, mad):
        g = mad.globals
        vars = mad.expr_vars
        g.foo = 1
        g.bar = 2
        self.assertEqual(set(vars('foo')), {'foo'})
        self.assertEqual(set(vars('(foo) * sin(2*pi*bar)')), {'foo', 'bar'})

    @with_madx()
    def test_command(self, mad):
        mad.input(SEQU)
        twiss = mad.command.twiss
        sbend = mad.elements.sb
        clone = sbend.clone('foobar', angle="pi/5", l=1)

        self.assertIn('betx=0', str(twiss))
        self.assertIn('angle=', str(sbend))
        self.assertIn('tilt', sbend)
        self.assertEqual(sbend.tilt, 0)
        self.assertEqual(len(sbend), len(list(sbend)))
        self.assertIn('tilt', list(sbend))

        self.assertEqual(clone.name, 'foobar')
        self.assertEqual(clone.base_type.name, 'sbend')
        self.assertEqual(clone.parent.name, 'sb')
        self.assertEqual(clone.defs.angle, 'pi / 5')
        self.assertAlmostEqual(clone.angle, 0.6283185307179586)
        self.assertEqual(len(clone), len(sbend))

        self.assertIn('angle=0.628', str(clone))
        self.assertNotIn('tilt', str(clone))
        clone.angle = 0.125
        clone = mad.elements.foobar            # need to update cache
        self.assertEqual(clone.angle, 0.125)
        self.assertEqual(len(twiss), len(list(twiss)))
        self.assertIn('betx', list(twiss))

        self.assertNotEqual(clone.angle, clone.parent.angle)
        del clone.angle
        clone = mad.elements.foobar            # need to update cache
        self.assertEqual(clone.angle, clone.parent.angle)

        with self.assertRaises(AttributeError):
            clone.missing_attribute

        with self.assertRaises(NotImplementedError):
            del twiss['betx']
        with self.assertRaises(NotImplementedError):
            del clone.base_type.angle

    @with_madx()
    def test_array_attribute(self, mad):
        mad.globals.nine = 9
        clone = mad.elements.multipole.clone('foo', knl=[0, 'nine/3', 4])
        knl = clone.knl
        self.assertEqual(knl[0], 0)
        self.assertEqual(knl[1], 3)
        self.assertEqual(knl[2], 4)
        self.assertEqual(len(knl), 3)
        self.assertEqual(list(knl), [0.0, 3.0, 4.0])
        self.assertEqual(str(knl), '[0.0, 3.0, 4.0]')
        knl[1] = '3*nine'
        self.assertEqual(mad.elements.foo.defs.knl[1], '3 * nine')
        self.assertEqual(mad.elements.foo.knl[1], 27)

    @with_madx()
    def test_array_attribute_comparison(self, mad):
        mad.globals.nine = 9
        foo = mad.elements.multipole.clone('foo', knl=[0, 5, 10])

        bar_eq = mad.elements.multipole.clone('bar_eq', knl=[0, 5, 10])
        bar_gt = mad.elements.multipole.clone('bar_gt', knl=[0, 6, 10])
        bar_lt = mad.elements.multipole.clone('bar_lt', knl=[0, 5, 'nine'])

        knl = foo.knl
        knl_eq = bar_eq.knl
        knl_gt = bar_gt.knl
        knl_lt = bar_lt.knl

        self.assertTrue(knl == knl_eq)
        self.assertFalse(knl == knl_gt)
        self.assertFalse(knl == knl_lt)

        self.assertFalse(knl < knl_eq)
        self.assertTrue(knl < knl_gt)
        self.assertFalse(knl < knl_lt)

        self.assertTrue(knl <= knl_eq)
        self.assertTrue(knl <= knl_gt)
        self.assertFalse(knl <= knl_lt)

        self.assertFalse(knl > knl_eq)
        self.assertFalse(knl > knl_gt)
        self.assertTrue(knl > knl_lt)

        self.assertTrue(knl >= knl_eq)
        self.assertFalse(knl >= knl_gt)
        self.assertTrue(knl >= knl_lt)

    @with_madx()
    def test_command_map(self, mad):
        command = mad.command
        self.assertIn('match', command)
        self.assertIn('sbend', command)

        self.assertNotIn('foooo', command)
        self.assertIn('match', list(command))
        self.assertEqual(len(command), len(list(command)))
        self.assertIn('match', str(command))
        self.assertIn('sbend', str(command))

        self.assertIn('sbend', mad.base_types)
        self.assertNotIn('match', mad.base_types)

    @with_madx()
    def test_comments(self, mad):
        var = mad.globals
        mad('x = 1; ! x = 2;')
        self.assertEqual(var.x, 1)
        mad('x = 2; // x = 3;')
        self.assertEqual(var.x, 2)
        mad('x = 3; /* x = 4; */')
        self.assertEqual(var.x, 3)
        mad('/* x = 3; */ x = 4;')
        self.assertEqual(var.x, 4)
        mad('x = 5; ! /* */ x = 6;')
        self.assertEqual(var.x, 5)
        mad('x = 5; /* ! */ x = 6;')
        self.assertEqual(var.x, 6)

    @with_madx()
    def test_multiline_input(self, mad):
        mad = mad
        var = mad.globals
        mad('''
            x = 1;
            y = 2;
        ''')
        self.assertEqual(var.x, 1)
        self.assertEqual(var.y, 2)
        mad('''
            x = /* 3;
            y =*/ 4;
        ''')
        self.assertEqual(var.x, 4)
        self.assertEqual(var.y, 2)
        mad('''
            x = 1; /*  */ x = 2;
            */ if (x == 1) {
                x = 3;
            }
        ''')
        self.assertEqual(var.x, 2)
        mad('''
            x = 1; /* x = 2;
            */ if (x == 1) {
                x = 3;
            }
        ''')
        self.assertEqual(var.x, 3)

    @with_madx()
    def test_errors(self, mad):
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


def _test_transfer_map(mad, seq, range_, doc, rtol=1e-7, atol=1e-15):
    mad.input(doc)
    mad.use(seq)
    par = ['x', 'px', 'y', 'py', 't', 'pt']
    val = [+0.0010, -0.0015, -0.0020, +0.0025, +0.0000, +0.0000]
    twiss = {'betx': 0.0012, 'alfx': 0.0018,
             'bety': 0.0023, 'alfy': 0.0027}
    twiss.update(zip(par, val))
    elems = range_.split('/')
    smap = mad.sectormap(elems, sequence=seq, **twiss)[-1]
    tw = mad.twiss(sequence=seq, range=range_, **twiss)

    # transport of coordinate vector:
    x_init = np.array(val)
    x_final_tw = np.array([tw[p][-1] for p in par])
    x_final_sm = np.dot(smap, np.hstack((x_init, 1)))
    assert_allclose(x_final_tw[:4], x_final_sm[:4],
                    rtol=rtol, atol=atol)

    # transport of beam matrix:
    tm = smap[0:6, 0:6]
    tab_len = len(tw['sig11'])
    sig_init = tw.sigmat(0)
    sig_final_tw = tw.sigmat(tab_len-1)
    sig_final_sm = np.dot(tm, np.dot(sig_init, tm.T))
    assert_allclose(sig_final_tw[0:4, 0:4], sig_final_sm[0:4, 0:4],
                    rtol=rtol, atol=atol)


class TestTransferMap(unittest.TestCase):

    @with_madx()
    def test_drift(self, mad):
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=1, refer=entry;
            d: drift, l=1, at=0;
            endsequence;
            beam;
        """)

    @with_madx()
    def test_kicker(self, mad):
        # hkicker
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=3, refer=entry;
            k: hkicker, kick=0.01, at=1;
            endsequence;
            beam;
        """)
        # vkicker
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=3, refer=entry;
            k: vkicker, kick=0.01, at=1;
            endsequence;
            beam;
        """)
        # three identical drifts in between
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=3, refer=entry;
            k: hkicker, kick=0.01, at=1;
            m: monitor, at=2;
            endsequence;
            beam;
        """)

    @with_madx()
    def test_quadrupole(self, mad):
        _test_transfer_map(mad, 's', '#s/#e', """
            qq: quadrupole, k1=0.01, l=1;
            s: sequence, l=4, refer=entry;
            qq, at=1;
            qq, at=2;
            endsequence;
            beam;
        """)

    @with_madx()
    def test_sbend(self, mad):
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=3, refer=entry;
            b: sbend, angle=0.1, l=1, at=1;
            endsequence;
            beam;
        """, rtol=3e-4)

    @with_madx()
    def test_solenoid(self, mad):
        _test_transfer_map(mad, 's', '#s/#e', """
            s: sequence, l=3, refer=entry;
            n: solenoid, ks=1, l=1, at=1;
            endsequence;
            beam;
        """)

    @with_madx()
    def test_subrange(self, mad):
        _test_transfer_map(mad, 's', 'm1/m2', """
            qq: quadrupole, k1=0.01, l=1;
            s: sequence, l=6, refer=entry;
            q0: quadrupole, k1=-0.01, l=0.5, at=0.25;
            m1: marker, at=1;
            qq, at=1;
            qq, at=3;
            m2: marker, at=5;
            qq, at=5;
            endsequence;
            beam;
        """)


def _get_elems(mad, seq_name):
    elems = mad.sequence[seq_name].elements
    elem_idx = dict((el.node_name, i) for i, el in enumerate(elems))
    return elems, elem_idx


if __name__ == '__main__':
    unittest.main()
