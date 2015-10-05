# standard library
import os
import sys
import unittest

# utilities
import _compat

# tested class
from cpymad.madx import Madx, CommandLog


class TestMadx(unittest.TestCase, _compat.TestCase):

    """
    Test methods for the Madx class.

    The tests are directly based on the specifics of the sequence in

        test/data/lebt/init.madx

    Please compare this file for reference.
    """

    def setUp(self):
        self.mad = Madx(command_log=CommandLog(sys.stdout, 'X:> '))
        here = os.path.dirname(__file__)
        there = os.path.join(here, 'data', 'lebt', 'init.madx')
        self.doc = open(there).read()
        for line in self.doc.splitlines():
            self.mad._libmadx.input(line)

    def tearDown(self):
        del self.mad

    def test_version(self):
        """Check that the Madx.version attribute can be used as expected."""
        version = self.mad.version
        # check format:
        major, minor, mini = map(int, version.release.split('.'))
        # We need at least MAD-X 5.02.03:
        self.assertGreaterEqual(major, 5)
        self.assertGreaterEqual(minor, 2)
        self.assertGreaterEqual(mini, 3)
        # check format:
        year, month, day = map(int, version.date.split('.'))
        self.assertGreaterEqual(year, 2014)
        self.assertGreaterEqual(month, 1)
        self.assertGreaterEqual(day, 1)
        self.assertLessEqual(month, 12)
        self.assertLessEqual(day, 31)

    def test_independent_instances(self):
        # create a second Madx instance (1st one is created in setUp)
        madxness = Madx()
        # Check independence by defining a variable differently in each
        # instance:
        self.mad.input('ANSWER=42;')
        madxness.input('ANSWER=43;')
        self.assertEqual(self.mad.evaluate('ANSWER'), 42);
        self.assertEqual(madxness.evaluate('ANSWER'), 43);

    def test_command_log(self):
        """Check that the command log contains all input commands."""
        # create a new Madx instance that uses the history feature:
        history_filename = '_test_madx.madx.tmp'
        mad = Madx(command_log=history_filename)
        # feed some input and compare with history file:
        for line in self.doc.splitlines():
            mad.input(line)
        with open(history_filename) as history_file:
            history = history_file.read()
        self.assertEqual(history.strip(), self.doc.strip())
        # remove history file
        del mad
        os.remove(history_filename)

    # TODO:
    # def test_command(self):
    # def test_help(self):
    # def test_call(self):

    def _check_twiss(self, seq_name):
        beam = 'beam, ex=1, ey=2, particle=electron, sequence={0};'.format(seq_name)
        self.mad.command(beam)
        initial = dict(alfx=0.5, alfy=1.5,
                       betx=2.5, bety=3.5)
        # by explicitly specifying the 'columns' parameter a persistent copy
        # is returned. We check that this copy contains the data we want and
        # that it has a 'summary' attribute:
        twiss = self.mad.twiss(sequence=seq_name,
                               columns=['betx', 'bety', 'alfx', 'alfy'],
                               **initial)
        betx, bety = twiss['betx'], twiss['bety']
        alfx, alfy = twiss['alfx'], twiss['alfy']
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

    def test_twiss_1(self):
        self._check_twiss('s1')     # s1 can be computed at start
        self._check_twiss('s1')     # s1 can be computed multiple times
        self._check_twiss('s2')     # s2 can be computed after s1

    def test_twiss_2(self):
        self._check_twiss('s2')     # s2 can be computed at start
        self._check_twiss('s1')     # s1 can be computed after s2

    def test_twiss_with_range(self):
        beam = 'beam, ex=1, ey=2, particle=electron, sequence=s1;'
        self.mad.command(beam)
        params = dict(alfx=0.5, alfy=1.5,
                      betx=2.5, bety=3.5,
                      columns=['betx', 'bety'],
                      sequence='s1')
        # Compute TWISS on full sequence, then on a sub-range, then again on
        # the full sequence. This checks that none of the range selections
        # have side-effects on each other:
        betx_full1 = self.mad.twiss(**params)['betx']
        betx_range = self.mad.twiss(range=('dr[2]', 'sb'), **params)['betx']
        betx_full2 = self.mad.twiss(**params)['betx']
        # Check that the results have the expected lengths:
        self.assertEqual(len(betx_full1), 9)
        self.assertEqual(len(betx_range), 4)
        self.assertEqual(len(betx_full2), 9)
        # Check numeric results. Since the first 3 elements of range and full
        # sequence are identical, equal results are expected. And non-equal
        # results afterwards.
        self.assertAlmostEqual(betx_range[0], betx_full1[1]) # dr:2, dr:1
        self.assertAlmostEqual(betx_range[1], betx_full1[2]) # qp:2, qp:1
        self.assertAlmostEqual(betx_range[2], betx_full1[3]) # dr:3, dr:2
        self.assertNotAlmostEqual(betx_range[3], betx_full1[4]) # sb, qp:2

    def test_range_row_api(self):
        beam = 'beam, ex=1, ey=2, particle=electron, sequence=s1;'
        self.mad.command(beam)
        params = dict(alfx=0.5, alfy=1.5,
                      betx=2.5, bety=3.5,
                      sequence='s1')
        tab = self.mad.twiss(range=('dr[2]', 'sb'), **params)
        self.assertEqual(tab.range, ('dr[2]', 'sb'))
        self.assertIn('betx', tab)

    # def test_survey(self):
    # def test_aperture(self):
    # def test_use(self):
    # def test_match(self):
    # def test_verbose(self):

    def test_active_sequence(self):
        self.mad.command('beam, ex=1, ey=2, particle=electron, sequence=s1;')
        self.mad.active_sequence = 's1'
        self.assertEqual(self.mad.active_sequence.name, 's1')

    def test_get_sequence(self):
        with self.assertRaises(KeyError):
            self.mad.sequences['sN']
        s1 = self.mad.sequences['s1']
        self.assertEqual(s1.name, 's1')

    def test_get_sequences(self):
        seqs = self.mad.sequences
        self.assertItemsEqual(seqs, ['s1', 's2'])

    def test_evaluate(self):
        val = self.mad.evaluate("1/QP_K1")
        self.assertAlmostEqual(val, 0.5)

    def test_set_value(self):
        self.mad.set_value('FOO', 1)
        self.mad.set_value('BAR', 'FOO')
        self.mad.set_value('FOO', 2)
        self.assertEqual(self.mad.evaluate('FOO'), 2)
        self.assertEqual(self.mad.evaluate('BAR'), 1)

    def test_set_expression(self):
        self.mad.set_expression('FOO', 'BAR')
        self.mad.set_value('BAR', 42)
        self.mad.evaluate('FOO')
        self.assertEqual(self.mad.evaluate('FOO'), 42)
        self.mad.set_value('BAR', 43)
        self.assertEqual(self.mad.evaluate('FOO'), 43)

    def test_globals(self):
        self.assertNotIn('FOO', self.mad.globals)
        self.mad.globals['FOO'] = 2
        self.assertIn('FOO', self.mad.globals)
        self.assertEqual(self.mad.globals['FOO'], 2)
        self.assertEqual(self.mad.evaluate('FOO'), 2)
        self.mad.globals['FOO'] = 3
        self.assertEqual(self.mad.evaluate('FOO'), 3)

    def test_elements(self):
        self.assertIn('sb', self.mad.elements)
        self.assertIn('sb', list(self.mad.elements))
        self.assertNotIn('foobar', self.mad.elements)
        self.assertAlmostEqual(self.mad.elements['sb']['angle'], 3.14/4)
        idx = self.mad.elements.index('qp1')
        elem = self.mad.elements[idx]
        self.assertEqual(elem['k1'], 3)

    # def test_sequence_beam(self):
    # def test_sequence_twiss(self):
    # def test_sequence_twissname(self):

    def _get_elems(self, seq_name):
        elems = self.mad.sequences[seq_name].elements
        elem_idx = dict((el['name'], i) for i, el in enumerate(elems))
        return elems, elem_idx

    def test_sequence_get_elements_s1(self):
        s1, idx = self._get_elems('s1')
        qp1 = s1['qp[1]']
        qp2 = s1['qp[2]']
        sb1 = s1['sb[1]']
        self.assertLess(idx['qp'], idx['qp[2]'])
        self.assertLess(idx['qp[2]'], idx['sb'])
        self.assertAlmostEqual(qp1['at'], 1)
        self.assertAlmostEqual(qp2['at'], 3)
        self.assertAlmostEqual(sb1['at'], 5)
        self.assertAlmostEqual(qp1['l'], 1)
        self.assertAlmostEqual(qp2['l'], 1)
        self.assertAlmostEqual(sb1['l'], 2)
        self.assertAlmostEqual(float(qp1['k1']), 2)
        self.assertAlmostEqual(float(qp2['k1']), 2)
        self.assertAlmostEqual(float(sb1['angle']), 3.14/4)
        self.assertEqual(str(qp1['k1']).lower(), "qp_k1")

    def test_sequence_get_elements_s2(self):
        s2, idx = self._get_elems('s2')
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

    def test_crash(self):
        """Check that a RuntimeError is raised in case MAD-X crashes."""
        # a.t.m. MAD-X crashes on this input, because the L (length)
        # parametere is missing:
        self.assertRaises(RuntimeError, self.mad.input, 'XXX: sequence;')

    def test_sequence_elements(self):
        elements = self.mad.sequences['s1'].elements
        iqp2 = elements.index('qp[2]')
        qp1 = elements['qp[1]']
        qp2 = elements[iqp2]
        self.assertAlmostEqual(qp1['at'], 1)
        self.assertAlmostEqual(qp2['at'], 3)
        self.assertEqual(iqp2, elements.at(3.1))

    def test_sequence_expanded_elements(self):
        beam = 'beam, ex=1, ey=2, particle=electron, sequence=s1;'
        self.mad.command(beam)
        self.mad.use('s1')
        elements = self.mad.sequences['s1'].expanded_elements
        iqp2 = elements.index('qp[2]')
        qp1 = elements['qp[1]']
        qp2 = elements[iqp2]
        self.assertAlmostEqual(qp1['at'], 1)
        self.assertAlmostEqual(qp2['at'], 3)
        self.assertEqual(iqp2, elements.at(3.1))


if __name__ == '__main__':
    unittest.main()
