from __future__ import print_function, division

import unittest
import numpy as np
from ImageD11 import friedel_stats, transform, cImageD11

L = 2e5   # sample to detector, um


def spot(g, t, pars, near=None, rng=None):
    """ lab position and omega of the spot of g for a grain at t (a detector
    sphere of radius L around the grain): the omega solution nearest near """
    tth, eta, om = transform.uncompute_g_vectors(g[:, None], pars['wavelength'], wedge=pars['wedge'], chi=pars['chi'])
    if not tth[0] > 0:
        return None
    if near is None:
        j = rng.randint(2)
    else:
        j = int(np.argmin([abs((o[0] - near + 180) % 360 - 180) for o in om]))
    o = om[j][0]
    tt, ee = np.radians(tth[0]), np.radians(eta[j][0])
    k = np.array([np.cos(tt), -np.sin(tt) * np.sin(ee), np.sin(tt) * np.cos(ee)])
    c, s = np.cos(np.radians(o)), np.sin(np.radians(o))
    origin = friedel_stats._axis_matrix(pars).dot([c * t[0] - s * t[1], s * t[0] + c * t[1], t[2]])
    return origin + L * k, o


def pairs_for_grain(t, pars, n, rng):
    """ n Friedel pairs (spot of g, spot of -g near omega + 180) of a grain at t """
    x1, o1, x2, o2, gs = [], [], [], [], []
    while len(gs) < n:
        g = rng.normal(size=3)
        g *= rng.uniform(0.3, 1.0) / np.linalg.norm(g)
        a = spot(g, t, pars, rng=rng)
        if a is None:
            continue
        b = spot(-g, t, pars, near=a[1] + 180)
        if b is None:
            continue
        x1.append(a[0]); o1.append(a[1]); x2.append(b[0]); o2.append(b[1]); gs.append(g)
    return np.array(x1), np.array(o1), np.array(x2), np.array(o2), np.array(gs)


class TestGeometry(unittest.TestCase):
    def test_gv_rows_matches_compute_gv(self):
        rng = np.random.RandomState(0)
        xyz = np.column_stack([np.full(500, L), rng.uniform(-5e4, 5e4, (500, 2))])
        om = rng.uniform(-180, 180, 500)
        t = np.array([120., -340., 55.])
        for wedge, chi, sign in ((0, 0, 1), (0.35, 0.1, 1), (-0.2, 0, -1)):
            pars = dict(wedge=wedge, chi=chi, omegasign=float(sign), wavelength=0.3)
            ref = np.empty((500, 3))
            cImageD11.compute_gv(xyz, om, float(sign), 0.3, float(wedge), float(chi), t, ref)
            self.assertLess(np.abs(friedel_stats.gv_rows(xyz, om, t, pars) - ref).max(), 1e-12)

    def test_pair_equation(self):
        # forward model: g_s1(t) + g_s2(t) = 0 at the grain's t, also with wedge
        rng = np.random.RandomState(1)
        t = np.array([300., -200., 50.])
        for wedge in (0.0, 0.35):
            pars = dict(wedge=wedge, chi=0.0, omegasign=1.0, wavelength=0.25)
            x1, o1, x2, o2, g = pairs_for_grain(t, pars, 20, rng)
            r = friedel_stats._pair_residual(x1, o1, x2, o2, t, pars)
            self.assertLess(np.abs(r).max(), 1e-12)


class TestPositions(unittest.TestCase):
    def test_fit_position(self):
        rng = np.random.RandomState(2)
        for wedge in (0.0, 0.35):
            pars = dict(wedge=wedge, chi=0.0, omegasign=1.0, wavelength=0.25)
            t = rng.uniform(-500, 500, 3)
            x1, o1, x2, o2, g = pairs_for_grain(t, pars, 30, rng)
            x1 = x1 + rng.normal(scale=20., size=x1.shape)   # spot centroid noise
            tfit, rms, n = friedel_stats.fit_position(x1, o1, x2, o2, pars)
            self.assertLess(np.abs(tfit - t).max(), 20.)


class TestRingWindows(unittest.TestCase):
    def test_width_and_offset(self):
        rng = np.random.RandomState(3)
        ringds = np.array([0.5, 0.7, 0.9])
        ds = np.concatenate([r + 2e-5 + 1e-4 * rng.normal(size=5000) for r in ringds] +
                            [rng.uniform(0.4, 1.0, 3000)])
        W = friedel_stats.ring_windows(ds, ringds, [0, 1, 2])
        for r in range(3):
            m, s, f, w = W[r]
            self.assertLess(abs(m - 2e-5), 1e-5)
            self.assertLess(abs(s - 1e-4) / 1e-4, 0.1)
            self.assertGreater(w, 3 * s)


if __name__ == '__main__':
    unittest.main()
