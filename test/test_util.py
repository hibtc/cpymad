"""
Tests for the functionality in :mod:`cpymad.util`.
"""

from cpymad import util
from cpymad.madx import Madx, AttrDict
from cpymad.types import Range, Constraint
import pytest


@pytest.fixture
def mad():
    with Madx(prompt='X:> ') as mad:
        yield mad


def is_valid_expression(expr):
    try:
        return util.check_expression(expr)
    except ValueError:
        return False


def test_mad_quote():
    # default to double quotes:
    assert util.mad_quote('foo bar') == '"foo bar"'
    # fallback to single quotes:
    assert util.mad_quote('"foo bar"') == '\'"foo bar"\''
    # MAD-X doesn't know escapes:
    with pytest.raises(ValueError):
        util.mad_quote('"foo" \'bar\'')


def test_is_identifier():
    assert util.is_identifier('hdr23_oei')
    assert util.is_identifier('_hdr23_oei')
    assert not util.is_identifier('2_hdr23_oei')
    assert not util.is_identifier('hdr oei')
    assert not util.is_identifier('hdr@oei')


def test_expr_symbols():
    assert util.expr_symbols('foobar') == {'foobar'}
    assert util.expr_symbols('foo*bar') == {'foo', 'bar'}
    assert util.expr_symbols('quad->k1') == {'quad->k1'}
    assert util.expr_symbols('q->k-p->k') == {'q->k', 'p->k'}
    assert util.expr_symbols('a * sin(x)') == {'a', 'sin', 'x'}


def test_name_from_internal():
    assert util.name_from_internal('foo.23:1') == 'foo.23'
    assert util.name_from_internal('foo.23:43') == 'foo.23[43]'
    with pytest.raises(ValueError):
        util.name_from_internal('foo:23o')


def test_name_to_internal():
    assert util.name_to_internal('foo.23') == 'foo.23:1'
    assert util.name_to_internal('foo.23[43]') == 'foo.23:43'
    with pytest.raises(ValueError):
        util.name_to_internal('foo:23o')


def test_normalize_range_name():
    assert util.normalize_range_name('dr[1]') == 'dr[1]'
    assert util.normalize_range_name('lebt$start') == '#s'
    assert util.normalize_range_name('lebt$end') == '#e'
    assert util.normalize_range_name('dr') == 'dr'
    assert util.normalize_range_name('l$start/qp[5]') == '#s/qp[5]'


def test_format_command(mad):
    assert util.format_command(
        'twiss', sequence='lhc'
    ) == 'twiss, sequence="lhc";'
    assert util.format_command(
        'option', echo=True
    ) == 'option, echo=true;'
    assert util.format_command(
        'constraint', betx=Constraint(max=3.13)
    ) == 'constraint, betx<3.13;'
    assert util.format_command(
        mad.command.quadrupole, k1='pi/2'
    ) == 'quadrupole, k1:=pi/2;'
    assert util.format_command(
        'multipole', knl=[0.0, 1.0, 2.0]
    ) == 'multipole, knl={0.0,1.0,2.0};'
    assert util.format_command(
        'twiss', range=Range('#s', '#e')
    ) == 'twiss, range=#s/#e;'
    assert util.format_command(
        'select', class_='quadrupole',
    ) == 'select, class="quadrupole";'

    assert util.format_command(
        mad.elements.quadrupole, k1="hello + world"
    ) == 'quadrupole, k1:=hello + world;'
    assert util.format_command(
        # match->sequence parameter is list in MAD-X!
        mad.command.match, sequence="foo"
    ) == "match, sequence=foo;"
    assert util.format_command(
        mad.command.select, class_='quadrupole',
    ) == 'select, class=quadrupole;'


def test_format_param():
    fmt = util.format_param

    assert fmt('foo', None) is None

    assert fmt('foo', Range('#s', '#e')) == 'foo=#s/#e'

    assert fmt('foo', Constraint(min="bar")) == 'foo>bar'
    assert fmt('foo', Constraint(max="bar")) == 'foo<bar'
    assert fmt('foo', Constraint("bar"))     == 'foo=bar'
    assert fmt('foo', Constraint("bar", min=-2, max=3)) == 'foo>-2, foo<3'

    assert fmt('foo', True) == 'foo=true'
    assert fmt('foo', False) == 'foo=false'

    assert fmt('range', 'l$start/l$end') == 'range=#s/#e'
    assert fmt('range', 'qp[5]') == 'range=qp[5]'
    assert fmt('range', 'qp[5]') == 'range=qp[5]'
    assert fmt('range', ('l$start', 'qp[1]')) == 'range=#s/qp[1]'

    assert fmt('file', "Weird N\ame!") == 'file="Weird N\ame!"'
    assert fmt('fool', "Weird N\ame!") == 'fool="weird n\ame!"'

    assert fmt('knl', [1.5, 'k1', 'k2']) == 'knl={1.5,k1,k2}'

    assert fmt('pi', 3) == 'pi=3'
    assert fmt('pi', 3.14) == 'pi=3.14'


def test_format_cmdpar(mad):
    fmt = util.format_cmdpar

    mult = mad.elements.quadrupole
    cons = mad.command.constraint
    match = mad.command.match
    twiss = mad.command.twiss

    assert fmt(mult, 'knl', None) == ''

    assert fmt(twiss, 'range', 'h$start/h$end') == 'range=#s/#e'
    assert fmt(twiss, 'range', ('a', 'b')) == 'range=a/b'
    assert fmt(match, 'range', '#s/#e') == 'range=#s/#e'
    assert fmt(match, 'range', ['#s/#e', 's/e']) == 'range={#s/#e,s/e}'

    assert fmt(cons, 'betx', Constraint(min="bar")) == 'betx>bar'
    assert fmt(cons, 'betx', Constraint(max="bar")) == 'betx<bar'
    assert fmt(cons, 'betx', Constraint(max="bar")) == 'betx<bar'
    assert fmt(
        cons, 'betx', Constraint("bar", min=-2, max=3)
    ) == 'betx>-2, betx<3'
    assert fmt(cons, 'betx', Constraint("bar")) == 'betx=bar'
    assert fmt(cons, 'betx', "bar")             == 'betx:=bar'
    assert fmt(cons, 'betx', 3.14)              == 'betx=3.14'

    assert fmt(match, 'slow', True) == 'slow=true'
    assert fmt(match, 'slow', False) == 'slow=false'

    assert fmt(twiss, 'file', "Weird N\ame!") == 'file="Weird N\ame!"'

    assert fmt(mult, 'knl', [1.5, 'x1', 'x2']) == 'knl:={1.5,x1,x2}'
    assert fmt(mult, 'knl', [1.5, 2.5, 3.5]) == 'knl={1.5,2.5,3.5}'

    assert fmt(mult, 'kmin', 3) == 'kmin=3'
    assert fmt(mult, 'kmin', 3.14) == 'kmin=3.14'
    assert fmt(mult, 'kmin', "x/(b+c)") == 'kmin:=x/(b+c)'

    assert fmt(match, 'sequence', 'seq') == 'sequence=seq'
    assert fmt(match, 'sequence', ['s0', 's1']) == 'sequence={s0,s1}'

    # missing attributes:
    with pytest.raises(KeyError):
        fmt(mult, 'foo', None)


def test_check_expression():
    assert is_valid_expression('a*b')
    assert is_valid_expression('a * (qp->k1+7.) ^ .5e-3')
    assert is_valid_expression('-a*-3')
    assert is_valid_expression('E-m*c^2')
    assert is_valid_expression('-(x / y)')
    assert is_valid_expression('1 ^(2)')
    assert is_valid_expression('+ 1')
    assert is_valid_expression('x')
    assert is_valid_expression('1')
    assert is_valid_expression('1 + 2')
    assert is_valid_expression('sin(x)')
    assert is_valid_expression('1 + table(twiss, mqf, x)')

    assert not is_valid_expression('')
    assert not is_valid_expression('x x')
    assert not is_valid_expression('+')
    assert not is_valid_expression('*')
    assert not is_valid_expression('* 1')
    assert not is_valid_expression('1 +')
    assert not is_valid_expression('sin(x) +')
    assert not is_valid_expression('sin(x) + * ')

    assert not is_valid_expression('E=m*c^2')
    assert not is_valid_expression('(@)')
    assert not is_valid_expression('(()')
    assert not is_valid_expression('(+3')
    assert not is_valid_expression('1*')     # NOTE: valid in MAD-X
    assert not is_valid_expression('3+')     # NOTE: valid in MAD-X
    assert not is_valid_expression('()')
    assert not is_valid_expression('(1 | 2)')

    assert not is_valid_expression('1 2')
    assert not is_valid_expression('1 (2)')
    assert not is_valid_expression('^(2)')


def test_attrdict():
    pi = 3.14
    d = AttrDict({'foo': 'bar', 'pi': pi})
    assert d.foo == 'bar'
    assert d.pi == pi
    assert d['foo'] == 'bar'
    assert d['pi'] == pi
    assert len(d) == 2
    assert set(d) == {'foo', 'pi'}
    assert 'FoO' in d
    assert 'pI' in d
    assert 'foopi' not in d
    assert 'foo' in str(d)
    assert "'bar'" in str(d)
    assert 'pi' in str(d)
    assert '3.14' in str(d)
    with pytest.raises(AttributeError):
        d.foobar
    with pytest.raises(KeyError):
        d['foobar']
    with pytest.raises(AttributeError):
        del d.foo


# TODO: test other functions
