# standard library
import unittest
import os

# utilities
import _compat

# tested class
from cpymad.madx import Madx


class TestMadx(unittest.TestCase, _compat.TestCase):

    """Test methods for the Madx class."""

    def setUp(self):
        self.mad = Madx()
        here = os.path.dirname(__file__)
        there = os.path.join(here, 'data', 'lebt', 'init.madx')
        self.doc = open(there).read()
        for line in self.doc.splitlines():
            self.mad._libmadx.input(line)

    def tearDown(self):
        del self.mad

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

    # def test_survey(self):
    # def test_aperture(self):
    # def test_use(self):
    # def test_match(self):
    # def test_verbose(self):

    def test_active_sequence(self):
        self.mad.command('beam, ex=1, ey=2, particle=electron, sequence=s1;')
        self.mad.active_sequence = 's1'
        self.assertEqual(self.mad.active_sequence, 's1')

    def test_get_sequence(self):
        self.assertRaises(ValueError, self.mad.get_sequence, 'sN')
        s1 = self.mad.get_sequence('s1')
        self.assertEqual(s1.name, 's1')

    def test_get_sequences(self):
        seqs = self.mad.get_sequences()
        self.assertItemsEqual([seq.name for seq in seqs],
                              ['s1', 's2'])

    def test_get_sequence_names(self):
        self.assertItemsEqual(self.mad.get_sequence_names(),
                              ['s1', 's2'])

    def test_evaluate(self):
        val = self.mad.evaluate("1/QP_K1")
        self.assertAlmostEqual(val, 0.5)

    # def test_sequence_beam(self):
    # def test_sequence_twiss(self):
    # def test_sequence_twissname(self):

    def _get_elems(self, seq_name):
        elems = self.mad.get_sequence(seq_name).get_elements()
        elem_dict = dict((el['name'], el) for el in elems)
        elem_idx = dict((el['name'], i) for i, el in enumerate(elems))
        return elem_dict, elem_idx

    def test_sequence_get_elements_s1(self):
        s1, idx = self._get_elems('s1')
        qp1 = s1['qp:1']
        qp2 = s1['qp:2']
        sb1 = s1['sb:1']
        self.assertLess(idx['qp:1'], idx['qp:2'])
        self.assertLess(idx['qp:2'], idx['sb:1'])
        self.assertAlmostEqual(qp1['at'], 0)
        self.assertAlmostEqual(qp2['at'], 1)
        self.assertAlmostEqual(sb1['at'], 2)
        self.assertAlmostEqual(qp1['l'], 1)
        self.assertAlmostEqual(qp2['l'], 1)
        self.assertAlmostEqual(sb1['l'], 2)
        self.assertAlmostEqual(float(qp1['k1']), 2)
        self.assertAlmostEqual(float(qp2['k1']), 2)
        self.assertAlmostEqual(float(sb1['angle']), 3.14/4)
        self.assertEqual(str(qp1['k1']).lower(), "qp_k1")

    def test_sequence_get_elements_s2(self):
        s2, idx = self._get_elems('s2')
        qp1 = s2['qp1:1']
        qp2 = s2['qp2:1']
        self.assertLess(idx['qp1:1'], idx['qp2:1'])
        self.assertAlmostEqual(qp1['at'], 0)
        self.assertAlmostEqual(qp2['at'], 1)
        self.assertAlmostEqual(qp1['l'], 1)
        self.assertAlmostEqual(qp2['l'], 2)
        self.assertAlmostEqual(float(qp1['k1']), 3)
        self.assertAlmostEqual(float(qp2['k1']), 2)

    # def test_sequence_get_expanded_elements(self):

if __name__ == '__main__':
    unittest.main()
