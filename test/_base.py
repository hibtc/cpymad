import cern.cpymad.api as cpymad


# NOTE: Do not inherit from unittest.TestCase, otherwise unittest will try
# to invoke all the test_xxx methods which makes no sense for this base
# class.
class TestCpymad(object):

    def setUp(self):
        self.model = cpymad.load_model(self.name)
        self.model.madx.command.option(twiss_print=False)

    # It's a bit surprising that this doesn't happen by itself.. Hmmm...
    def tearDown(self):
        del self.model

    def test_twiss(self):
        twiss = self.model.twiss()
        self.assertTrue('betx' in twiss)
        self.assertTrue('bety' in twiss)
        self.assertTrue('s' in twiss)
        # check that keys are all lowercase..
        for k in twiss:
            self.assertEqual(k, k.lower())
        for k in twiss.summary:
            self.assertEqual(k, k.lower())

    def test_sequences(self):
        '''
         Checks that all sequences defined in the model (json)
         is also loaded into memory
        '''
        for seq in self.model.get_sequence_names():
            print('Testing set_sequence({0!r})'.format(seq))
            self.assertTrue(self.model.madx.has_sequence(seq))

    def test_set_optic(self):
        '''
         Sets all optics found in the model definition
        '''
        for optic in self.model.list_optics():
            print('Testing set_optic({0!r})'.format(optic))
            self.model.set_optic(optic)
            self.assertEqual(optic,self.model._active['optic'])
            self.model.twiss()

