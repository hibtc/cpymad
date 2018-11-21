# standard library
import unittest

# tested objects
from cpymad import util
from cpymad.madx import Madx, AttrDict
from cpymad.types import Range, Constraint


def is_valid_expression(expr):
    try:
        return util.check_expression(expr)
    except ValueError:
        return False


def _close(m):
    try:
        m._service.close()
    except (RuntimeError, IOError, EOFError, OSError):
        pass
    m._process.wait()


class TestUtil(unittest.TestCase):

    """Tests for the objects in :mod:`cpymad.util`."""

    def tearDown(self):
        if hasattr(self, 'madx'):
            _close(self.madx)

    def test_mad_quote(self):
        # default to double quotes:
        self.assertEqual(util.mad_quote('foo bar'), '"foo bar"')
        # fallback to single quotes:
        self.assertEqual(util.mad_quote('"foo bar"'), '\'"foo bar"\'')
        # MAD-X doesn't know escapes:
        with self.assertRaises(ValueError):
            util.mad_quote('"foo" \'bar\'')

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
        self.assertEqual(util.normalize_range_name('lebt$start'), '#s')
        self.assertEqual(util.normalize_range_name('lebt$end'), '#e')
        self.assertEqual(util.normalize_range_name('dr'), 'dr')
        self.assertEqual(util.normalize_range_name('l$start/qp[5]'), '#s/qp[5]')

    def _check_command(self, compare, *args, **kwargs):
        self.assertEqual(compare, util.format_command(*args, **kwargs))

    def test_format_command(self):
        m = self.madx = Madx()

        self.assertEqual(
            util.format_command(
                'twiss', sequence='lhc'
            ), 'twiss, sequence="lhc";')
        self.assertEqual(
            util.format_command(
                'option', echo=True
            ), 'option, echo=true;')
        self.assertEqual(
            util.format_command(
                'constraint', betx=Constraint(max=3.13)
            ), 'constraint, betx<3.13;')
        self.assertEqual(
            util.format_command(
                m.command.quadrupole, k1='pi/2'
            ), 'quadrupole, k1:=pi/2;')
        self.assertEqual(
            util.format_command(
                'multipole', knl=[0.0, 1.0, 2.0]
            ), 'multipole, knl={0.0,1.0,2.0};')
        self.assertEqual(
            util.format_command(
                'twiss', range=Range('#s', '#e')
            ), 'twiss, range=#s/#e;')
        self.assertEqual(
            util.format_command(
                'select', class_='quadrupole',
            ), 'select, class="quadrupole";')

        self.assertEqual(
            util.format_command(
                m.elements.quadrupole, k1="hello + world"
            ), 'quadrupole, k1:=hello + world;')
        self.assertEqual(
            util.format_command(
                # match->sequence parameter is list in MAD-X!
                m.command.match, sequence="foo"
            ), "match, sequence=foo;")
        self.assertEqual(
            util.format_command(
                m.command.select, class_='quadrupole',
            ), 'select, class=quadrupole;')

    def test_format_param(self):
        fmt = util.format_param

        self.assertEqual(fmt('foo', None), None)

        self.assertEqual(fmt('foo', Range('#s', '#e')), 'foo=#s/#e')

        self.assertEqual(fmt('foo', Constraint(min="bar")), 'foo>bar')
        self.assertEqual(fmt('foo', Constraint(max="bar")), 'foo<bar')
        self.assertEqual(fmt('foo', Constraint("bar")),     'foo=bar')
        self.assertEqual(fmt('foo', Constraint("bar", min=-2, max=3)),
                         'foo>-2, foo<3')

        self.assertEqual(fmt('foo', True), 'foo=true')
        self.assertEqual(fmt('foo', False), 'foo=false')

        self.assertEqual(fmt('range', 'l$start/l$end'), 'range=#s/#e')
        self.assertEqual(fmt('range', 'qp[5]'), 'range=qp[5]')
        self.assertEqual(fmt('range', 'qp[5]'), 'range=qp[5]')
        self.assertEqual(fmt('range', ('l$start', 'qp[1]')), 'range=#s/qp[1]')

        self.assertEqual(fmt('file', "Weird N\ame!"), 'file="Weird N\ame!"')
        self.assertEqual(fmt('fool', "Weird N\ame!"), 'fool="weird n\ame!"')

        self.assertEqual(fmt('knl', [1.5, 'k1', 'k2']), 'knl={1.5,k1,k2}')

        self.assertEqual(fmt('pi', 3), 'pi=3')
        self.assertEqual(fmt('pi', 3.14), 'pi=3.14')

    def test_format_cmdpar(self):
        fmt = util.format_cmdpar

        m = self.madx = Madx()
        mult = m.elements.quadrupole
        cons = m.command.constraint
        match = m.command.match
        twiss = m.command.twiss

        self.assertEqual(fmt(mult, 'knl', None), '')

        self.assertEqual(fmt(twiss, 'range', 'h$start/h$end'), 'range=#s/#e')
        self.assertEqual(fmt(twiss, 'range', ('a', 'b')), 'range=a/b')
        self.assertEqual(fmt(match, 'range', '#s/#e'), 'range=#s/#e')
        self.assertEqual(fmt(match, 'range', ['#s/#e', 's/e']),
                         'range={#s/#e,s/e}')

        self.assertEqual(fmt(cons, 'betx', Constraint(min="bar")), 'betx>bar')
        self.assertEqual(fmt(cons, 'betx', Constraint(max="bar")), 'betx<bar')
        self.assertEqual(fmt(cons, 'betx', Constraint(max="bar")), 'betx<bar')
        self.assertEqual(fmt(cons, 'betx', Constraint("bar", min=-2, max=3)),
                         'betx>-2, betx<3')
        self.assertEqual(fmt(cons, 'betx', Constraint("bar")),     'betx=bar')
        self.assertEqual(fmt(cons, 'betx', "bar"),                 'betx:=bar')
        self.assertEqual(fmt(cons, 'betx', 3.14),                  'betx=3.14')

        self.assertEqual(fmt(match, 'slow', True), 'slow=true')
        self.assertEqual(fmt(match, 'slow', False), 'slow=false')

        self.assertEqual(fmt(twiss, 'file', "Weird N\ame!"), 'file="Weird N\ame!"')

        self.assertEqual(fmt(mult, 'knl', [1.5, 'x1', 'x2']), 'knl:={1.5,x1,x2}')
        self.assertEqual(fmt(mult, 'knl', [1.5, 2.5, 3.5]), 'knl={1.5,2.5,3.5}')

        self.assertEqual(fmt(mult, 'kmin', 3), 'kmin=3')
        self.assertEqual(fmt(mult, 'kmin', 3.14), 'kmin=3.14')
        self.assertEqual(fmt(mult, 'kmin', "x/(b+c)"), 'kmin:=x/(b+c)')

        self.assertEqual(fmt(match, 'sequence', 'seq'), 'sequence=seq')
        self.assertEqual(fmt(match, 'sequence', ['s0', 's1']), 'sequence={s0,s1}')

        # missing attributes:
        with self.assertRaises(KeyError):
            fmt(mult, 'foo', None)

    def test_check_expression(self):
        self.assertTrue(is_valid_expression('a*b'))
        self.assertTrue(is_valid_expression('a * (qp->k1+7.) ^ .5e-3'))
        self.assertTrue(is_valid_expression('-a*-3'))
        self.assertTrue(is_valid_expression('E-m*c^2'))
        self.assertTrue(is_valid_expression('-(x / y)'))
        self.assertTrue(is_valid_expression('1 ^(2)'))

        self.assertFalse(is_valid_expression('E=m*c^2'))
        self.assertFalse(is_valid_expression('(@)'))
        self.assertFalse(is_valid_expression('(()'))
        self.assertFalse(is_valid_expression('(+3'))
        self.assertFalse(is_valid_expression('1*'))     # NOTE: valid in MAD-X
        self.assertFalse(is_valid_expression('3+'))     # NOTE: valid in MAD-X
        self.assertFalse(is_valid_expression('()'))
        self.assertFalse(is_valid_expression('(1 | 2)'))

        self.assertFalse(is_valid_expression('1 2'))
        self.assertFalse(is_valid_expression('1 (2)'))
        self.assertFalse(is_valid_expression('^(2)'))

    def test_attrdict(self):
        pi = 3.14
        d = AttrDict({'foo': 'bar', 'pi': pi})
        self.assertEqual(d.foo, 'bar')
        self.assertEqual(d.pi, pi)
        self.assertEqual(d['foo'], 'bar')
        self.assertEqual(d['pi'], pi)
        self.assertEqual(len(d), 2)
        self.assertEqual(set(d), {'foo', 'pi'})
        self.assertIn('FoO', d)
        self.assertIn('pI', d)
        self.assertNotIn('foopi', d)
        self.assertIn('foo', str(d))
        self.assertIn("'bar'", str(d))
        self.assertIn('pi', str(d))
        self.assertIn('3.14', str(d))
        with self.assertRaises(AttributeError):
            d.foobar
        with self.assertRaises(KeyError):
            d['foobar']
        with self.assertRaises(AttributeError):
            del d.foo

    # TODO: test other functions


if __name__ == '__main__':
    unittest.main()
