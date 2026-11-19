from __future__ import print_function, division

import os
import unittest
import numpy as np
from ImageD11 import grid_stats, grain, columnfile

HERE = os.path.dirname(os.path.abspath(__file__))


class TestRandomRotations(unittest.TestCase):
    def test_proper_rotations(self):
        R = grid_stats.random_rotations(100, np.random.RandomState(1))
        for r in R:
            self.assertTrue(np.allclose(r.dot(r.T), np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(r), 1.0)


class TestAccept(unittest.TestCase):
    def test_duplicates_and_chance(self):
        npk = 1000
        rng = np.random.RandomState(0)
        real = [np.arange(i * 100, i * 100 + 80) for i in range(5)]
        dup = [real[0][:70]]                      # re-uses a real grain's peaks
        junk = [rng.choice(npk, 5, replace=False) for _ in range(20)]   # below chance
        nulls = [rng.choice(npk, 10, replace=False) for _ in range(2000)]
        acc, claimed, cut, efalse, unclaimed, null_un = grid_stats.accept_grains(
            real + dup + junk, nulls, np.zeros(npk, bool), 1.0)
        self.assertEqual(sorted(acc), list(range(5)))
        self.assertLessEqual(efalse, 1.0)
        self.assertEqual(claimed.sum(), 400)


class TestRefineGrains(unittest.TestCase):
    def setUp(self):
        d = os.path.join(HERE, 'makemap')
        self.prm = os.path.join(d, 'test.prm')
        self.grains = grain.read_grain_file(os.path.join(d, 'map.ubi'))
        self.cf = columnfile.columnfile(os.path.join(d, 'test.flt'))

    def test_drops_failed_grain(self):
        # a fit that fails (left handed UBI) drops that grain only
        bad = self.grains[3].ubi.copy()
        refine = grid_stats._RefineDropFailed.refine

        def failing(o, ubi, quiet=True):
            if np.allclose(ubi, bad):
                raise AssertionError('Left handed axis system!')
            return refine(o, ubi, quiet)
        grid_stats._RefineDropFailed.refine = failing
        try:
            out, kept = grid_stats.refine_grains(self.prm, self.cf, self.grains, 0.02, 0.25, 'cubic')
        finally:
            grid_stats._RefineDropFailed.refine = refine
        self.assertEqual(kept, [i for i in range(len(self.grains)) if i != 3])
        self.assertEqual(len(out), len(self.grains) - 1)
        self.assertTrue((self.cf.labels >= 0).sum() > 0)


if __name__ == '__main__':
    unittest.main()
