# encoding: utf-8
"""
Tests for the model.Model runtime hierarchy.
"""

# tested classes
from cern.cpymad import model
from cern.resource.file import FileResource

# test utilities
import unittest
import os

__all__ = [
    'TestModel',
]


class TestModel(unittest.TestCase):

    path = os.path.join(os.path.dirname(__file__), 'data')
    name = 'lebt'

    def load_model(self):
        locator = model.Locator(FileResource(self.path))
        factory = model.Factory(locator)
        return factory(self.name)

    def setUp(self):
        self.model = self.load_model()
        self.model.madx.command.option(twiss_print=False)

    def tearDown(self):
        del self.model

    def test_twiss(self):
        seq = self.model.default_sequence
        twiss = seq.twiss()
        self.assertTrue('betx' in twiss)
        self.assertTrue('bety' in twiss)
        self.assertTrue('s' in twiss)
        # check that keys are all lowercase..
        for k in twiss:
            self.assertEqual(k, k.lower())
        for k in twiss.summary:
            self.assertEqual(k, k.lower())

    def test_sequences(self):
        """
        Checks that all sequences defined in the model (json)
        is also loaded into memory
        """
        for seq in self.model.sequences:
            print('Testing set_sequence({0!r})'.format(seq))
            self.assertTrue(self.model.madx.has_sequence(seq))

    def test_set_optic(self):
        """Sets all optics found in the model definition."""
        sequence = self.model.default_sequence
        for optic in self.model.optics.values():
            print('Testing set_optic({0!r})'.format(optic.name))
            optic.load()
            sequence.twiss()


if __name__ == '__main__':
    unittest.main()
