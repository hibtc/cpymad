# encoding: utf-8
"""
Tests for the model.Model runtime hierarchy.
"""

# tested classes
from cpymad.madx import CommandLog
from cpymad.model import Model
from cpymad.resource.file import FileResource

# utilities
import _compat

# standard library
import os
import sys
import unittest

__all__ = [
    'TestModel',
]


class TestModel(unittest.TestCase, _compat.TestCase):

    """
    Tests for the Model class.
    """

    # test configuration:

    path = os.path.join(os.path.dirname(__file__), 'data', 'lebt.cpymad.yml')

    # helper methods for tests:

    def load_model(self, path):
        """Load model with given name from specified path."""
        command_log = CommandLog(sys.stdout, 'X:> ')
        model = Model.load(self.path, command_log=command_log)
        model.madx.command.option(twiss_print=False)
        return model

    def setUp(self):
        self.model = self.load_model(self.path)

    def tearDown(self):
        del self.model

    # tests for Model API

    def test_compatibility_check(self):
        data = {
            'beams': {},
            'optics': {},
            'sequences': {},
        }
        with self.assertRaises(ValueError):
            Model(data=data, repo=None, madx=None)
        with self.assertRaises(ValueError):
            Model(data=dict(data, api_version=-1), repo=None, madx=None)
        with self.assertRaises(ValueError):
            Model(data=dict(data, api_version=1), repo=None, madx=None)
        Model(data=dict(data, api_version=0), repo=None, madx=None)

    def test_Model_API(self):
        """Check that the public Model attributes/methods behave reasonably."""
        model = self.model
        madx = model.madx
        # name
        self.assertEqual(model.name, 'lebt')
        # data
        repository = FileResource(self.path)
        self.assertEqual(model.data, repository.yaml())
        # beams
        self.assertItemsEqual(model.beams.keys(), ['carbon', 'other'])
        # optics
        self.assertItemsEqual(model.optics.keys(), ['thick'])
        # sequences
        self.assertItemsEqual(model.sequences.keys(), ['s1', 's2'])
        self.assertTrue('s1' in madx.sequences)
        self.assertTrue('s2' in madx.sequences)
        # default_optic
        self.assertIs(model.default_optic, model.optics['thick'])
        # default_sequence
        self.assertIs(model.default_sequence, model.sequences['s1'])

    # tests for Optic API

    def test_Optic_load(self):
        """Check that the Optic init-files are executed upon init()."""
        evaluate = self.model.madx.evaluate
        self.assertAlmostEqual(evaluate('NOT_ZERO'), 0.0)
        self.model.optics['thick'].init()
        self.assertAlmostEqual(evaluate('NOT_ZERO'), 1.0)

    # tests for Sequence API

    def test_Sequence_API(self):
        """Check that the general Sequence API behaves reasonable."""
        sequence = self.model.sequences['s1']
        # name
        self.assertEqual(sequence.name, 's1')
        # ranges
        self.assertItemsEqual(sequence.ranges.keys(), ['ALL'])
        # beam
        self.assertIs(sequence.beam, self.model.beams['carbon'])
        # default_range
        self.assertIs(sequence.default_range, sequence.ranges['ALL'])

    # def test_Sequence_twiss(self):        # see test_Optic_twiss for now
    # def test_Sequence_match(self):        # see test_Optic_match for now

    def test_Sequence_survey(self):
        """Execute survey() and check that it returns usable values."""
        seq = self.model.default_sequence
        survey = seq.survey()
        # access some data to make sure the table was generated:
        s = survey['s']
        x = survey['x']
        y = survey['y']
        z = survey['z']

    # tests for Range API

    def test_Range_API(self):
        """Check that the general Range API behaves reasonable."""
        range = self.model.default_sequence.default_range
        # name
        self.assertEqual(range.name, 'ALL')
        # bounds
        self.assertEqual(range.bounds, ('#s', '#e'))
        # initial_conditions
        self.assertItemsEqual(range.initial_conditions.keys(), ['default'])
        # default_initial_conditions
        self.assertIs(range.default_initial_conditions,
                      range.initial_conditions['default'])

    def test_Range_twiss(self):
        """Execute twiss() and check that it returns usable values."""
        range = self.model.default_sequence.default_range
        twiss = range.twiss()
        # access some data to make sure the table was generated:
        betx = twiss['betx']
        bety = twiss['bety']
        alfx = twiss['alfx']
        alfy = twiss['alfy']

    def test_Range_match(self):
        """Execute match() and check that it returns usable values."""
        range = self.model.sequences['s1'].default_range
        knobs = range.match(
            constraints=[dict(range='sb', betx=0.3)],
            vary=['QP_K1'],
        )
        knobs['QP_K1']


if __name__ == '__main__':
    unittest.main()
