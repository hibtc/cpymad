"""
Test that the returned transfer map can be used expected.
"""

import numpy as np
from numpy.testing import assert_allclose

from cpymad.madx import Madx


def test_drift():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=1, refer=entry;
        d: drift, l=1, at=0;
        endsequence;
        beam;
    """)


def test_hkicker():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=3, refer=entry;
        k: hkicker, kick=0.01, at=1;
        endsequence;
        beam;
    """)


def test_vkicker():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=3, refer=entry;
        k: vkicker, kick=0.01, at=1;
        endsequence;
        beam;
    """)


def test_drift_kick_drift():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=3, refer=entry;
        k: hkicker, kick=0.01, at=1;
        m: monitor, at=2;
        endsequence;
        beam;
    """)


def test_quadrupole():
    _test_transfer_map('s', '#s/#e', """
        qq: quadrupole, k1=0.01, l=1;
        s: sequence, l=4, refer=entry;
        qq, at=1;
        qq, at=2;
        endsequence;
        beam;
    """)


def test_sbend():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=3, refer=entry;
        b: sbend, angle=0.1, l=1, at=1;
        endsequence;
        beam;
    """, rtol=3e-4)


def test_solenoid():
    _test_transfer_map('s', '#s/#e', """
        s: sequence, l=3, refer=entry;
        n: solenoid, ks=1, l=1, at=1;
        endsequence;
        beam;
    """)


def test_subrange():
    _test_transfer_map('s', 'm1/m2', """
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


def _test_transfer_map(seq, range_, doc, rtol=1e-7, atol=1e-15):
    with Madx() as mad:
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
