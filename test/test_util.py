# standard library
import unittest

# tested objects
from cpymad import util
from cpymad.madx import Madx
from cpymad.types import Range, Constraint


def is_valid_expression(expr):
    try:
        return util.check_expression(expr)
    except ValueError:
        return False


class TestUtil(unittest.TestCase):

    """Tests for the objects in :mod:`cpymad.util`."""

    def test_is_identifier(self):
        self.assertTrue(util.is_identifier('hdr23_oei'))
        self.assertTrue(util.is_identifier('_hdr23_oei'))
        self.assertFalse(util.is_identifier('2_hdr23_oei'))
        self.assertFalse(util.is_identifier('hdr oei'))
        self.assertFalse(util.is_identifier('hdr@oei'))

    def test_expr_symbols(self):
        self.assertEqual(util.expr_symbols('foobar'), {'foobar'})
        self.assertEqual(util.expr_symbols('foo*bar'), {'foo', 'bar'})
        self.assertEqual(util.expr_symbols('quad->k1'), {'quad->k1'})
        self.assertEqual(util.expr_symbols('q->k-p->k'), {'q->k', 'p->k'})
        self.assertEqual(util.expr_symbols('a * sin(x)'), {'a', 'sin', 'x'})

    def test_name_from_internal(self):
        self.assertEqual(util.name_from_internal('foo.23:1'), 'foo.23')
        self.assertEqual(util.name_from_internal('foo.23:43'), 'foo.23[43]')
        with self.assertRaises(ValueError):
            util.name_from_internal('foo:23o')

    def test_name_to_internal(self):
        self.assertEqual(util.name_to_internal('foo.23'), 'foo.23:1')
        self.assertEqual(util.name_to_internal('foo.23[43]'), 'foo.23:43')
        with self.assertRaises(ValueError):
            util.name_to_internal('foo:23o')

    def test_normalize_range_name(self):
        self.assertEqual(util.normalize_range_name('dr[1]'), 'dr[1]')
        self.assertEqual(util.normalize_range_name('lebt$end'), '#e')
        self.assertEqual(util.normalize_range_name('dr'), 'dr')

    def _check_command(self, compare, *args, **kwargs):
        self.assertEqual(compare, util.format_command(*args, **kwargs))

    def test_format_command(self):
        m = Madx()

        self.assertEqual(
            util.format_command(
                'twiss', sequence='lhc'),
                'twiss, sequence="lhc";')
        self.assertEqual(
            util.format_command(
                'option', echo=True),
                'option, echo=true;')
        self.assertEqual(
            util.format_command(
                'constraint', betx=Constraint(max=3.13)),
                'constraint, betx<3.13;')
        self.assertEqual(
            util.format_command(
                m.command.quadrupole, k1='pi/2'),
                'quadrupole, k1:=pi/2;')
        self.assertEqual(
            util.format_command(
                'multipole', knl=[0.0, 1.0, 2.0]),
                'multipole, knl={0.0,1.0,2.0};')
        self.assertEqual(
            util.format_command(
                'twiss', range=Range('#s', '#e')),
                'twiss, range=#s/#e;')

        self.assertEqual(
            util.format_command(
                m.elements.quadrupole, k1="hello + world"),
                'quadrupole, k1:=hello + world;')
        self.assertEqual(
            util.format_command(
                # match->sequence parameter is list in MAD-X!
                m.command.match, sequence="foo"),
                "match, sequence=foo;")

    def test_check_expression(self):
        self.assertTrue(is_valid_expression('a*b'))
        self.assertTrue(is_valid_expression('a * (qp->k1+7.) ^ .5e-3'))
        self.assertTrue(is_valid_expression('-a*-3'))
        self.assertTrue(is_valid_expression('E-m*c^2'))

        self.assertFalse(is_valid_expression('E=m*c^2'))
        self.assertFalse(is_valid_expression('(@)'))
        self.assertFalse(is_valid_expression('(()'))
        self.assertFalse(is_valid_expression('(+3'))
        self.assertFalse(is_valid_expression('1*'))     # NOTE: valid in MAD-X
        self.assertFalse(is_valid_expression('3+'))     # NOTE: valid in MAD-X
        self.assertFalse(is_valid_expression('()'))
        self.assertFalse(is_valid_expression('(1 | 2)'))


    # TODO: test other functions


if __name__ == '__main__':
    unittest.main()
