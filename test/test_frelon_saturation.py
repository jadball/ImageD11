from __future__ import division, print_function
import unittest
import numpy as np
import scipy.ndimage
import ImageD11.cImageD11 as cid
from ImageD11.frelon_peaksearch import worker, peaks_list_to_dict, do3dmerge

SAT = 57000.0
SHAPE = (256, 256)


def make_worker(saturation):
    # background 100 ADU on the active area, 10 ADU (dark) on a 10 px border
    bg = np.full(SHAPE, 100, np.float32)
    bg[:10] = bg[-10:] = bg[:, :10] = bg[:, -10:] = 10
    mask = np.zeros(SHAPE, np.uint8)
    return worker(bg.copy(), mask, threshold=25, smoothsigma=1.0, bgc=0.9, minpx=3,
                  m_offset_thresh=50, m_ratio_thresh=50, saturation=saturation)


def bloom_frame():
    """A saturated spot bloomed along a column with two maxima, so that
    localmaxlabel splits it into two touching fragments, plus one weak spot."""
    rs = np.random.RandomState(42)
    img = np.full(SHAPE, 100, np.float32) + rs.normal(0, 2, SHAPE).astype(np.float32)
    img[:10] = img[-10:] = img[:, :10] = img[:, -10:] = 10
    rows = np.arange(80, 180)
    prof = 59000 + 2000 * (np.exp(-0.5 * ((rows - 100) / 6.) ** 2) + np.exp(-0.5 * ((rows - 160) / 6.) ** 2))
    img[80:180, 120:123] = prof[:, None]
    img[40:43, 40:43] += 500  # weak, unsaturated spot
    return img


class TestSaturation(unittest.TestCase):

    def test_no_saturation_unchanged(self):
        pks = make_worker(None).peaksearch(bloom_frame())
        self.assertEqual(pks.shape[1], cid.NPROPERTY)

    def test_nsat_counts_pixels(self):
        img = bloom_frame()
        pks = make_worker(SAT).peaksearch(img)
        self.assertEqual(pks.shape[1], cid.NPROPERTY + 1)
        nsat = pks[:, cid.NPROPERTY]
        self.assertEqual(nsat.sum(), (img >= SAT).sum())
        self.assertEqual((nsat > 0).sum(), 2)  # the bloom is split in two
        self.assertEqual((nsat == 0).sum(), 1)  # the weak spot

    def merged(self, saturation):
        w = make_worker(saturation)
        frames = [(i, w.peaksearch(bloom_frame(), omega=0.1 * i).copy()) for i in range(3)]
        d, n = peaks_list_to_dict(frames)
        d3 = do3dmerge(d, n, np.array([0.1 * i for i in range(3)]))
        return d, d3

    def test_bloom_fragments_linked(self):
        d, d3 = self.merged(None)
        self.assertEqual(len(d3['s_raw']), 3)  # two bloom fragments + weak spot
        d, d3 = self.merged(SAT)
        self.assertEqual(len(d3['s_raw']), 2)  # one bloom + weak spot
        self.assertEqual((d3['n_sat'] > 0).sum(), 1)
        self.assertEqual(d3['n_sat'].sum(), d['n_sat'].sum())
        # weak spot: distance to the bloom fragment centroids on its frames
        weak = d3['n_sat'] == 0
        self.assertTrue(np.isfinite(d3['sat_dist'][weak]).all())
        self.assertTrue((d3['sat_dist'][~weak] == 0).all())


class TestLocalmaxlabelThreads(unittest.TestCase):

    @unittest.skipIf(cid.cores_available() < 16,
                     "the race shows only with many threads (14/500 runs at 32, 0/500 at 4)")
    def test_threads_match_one_thread(self):
        # smoothed noise gives many basins with long uphill paths
        rs = np.random.RandomState(0)
        im = scipy.ndimage.gaussian_filter(rs.random_sample((2048, 2048)), 4).astype(np.float32)
        im -= np.median(im)
        im[im < 0] = 0
        lab = np.empty(im.shape, np.int32)
        wrk = np.empty(im.shape, np.int8)
        cid.cimaged11_omp_set_num_threads(1)
        n1 = cid.localmaxlabel(im, lab, wrk)
        ref = lab.copy()
        cid.cimaged11_omp_set_num_threads(min(32, cid.cores_available()))
        try:
            for _ in range(500):
                lab[:] = rs.randint(1, 10 ** 6, lab.shape)  # stale labels
                n = cid.localmaxlabel(im, lab, wrk)
                self.assertEqual(n, n1)
                self.assertTrue(np.array_equal(lab, ref))
        finally:
            cid.cimaged11_omp_set_num_threads(1)


if __name__ == '__main__':
    unittest.main()
