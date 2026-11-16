from __future__ import division, print_function
import unittest
import numpy as np
import scipy.ndimage
import ImageD11.cImageD11 as cid
from ImageD11.frelon_peaksearch import worker, peaks_list_to_dict, do3dmerge

SAT = 57000.0
SHAPE = (256, 256)


def make_worker(saturation, prominence=False, clipped=None):
    # background 100 ADU on the active area, 10 ADU (dark) on a 10 px border
    bg = np.full(SHAPE, 100, np.float32)
    bg[:10] = bg[-10:] = bg[:, :10] = bg[:, -10:] = 10
    mask = np.zeros(SHAPE, np.uint8)
    return worker(bg.copy(), mask, threshold=25, smoothsigma=1.0, bgc=0.9, minpx=3,
                  m_offset_thresh=50, m_ratio_thresh=50, saturation=saturation, prominence=prominence,
                  clipped=clipped)


def flat_frame(seed=42):
    rs = np.random.RandomState(seed)
    img = np.full(SHAPE, 100, np.float32) + rs.normal(0, 2, SHAPE).astype(np.float32)
    img[:10] = img[-10:] = img[:, :10] = img[:, -10:] = 10
    return img


def bloom_frame(clip=(0, 0, 0)):
    """A saturated spot bloomed along a column with two maxima, so that
    localmaxlabel splits it into two touching fragments, plus one weak spot.
    clip: offsets of the pile-up level in the three bloom columns."""
    img = flat_frame()
    rows = np.arange(80, 180)
    prof = 59000 + 2000 * (np.exp(-0.5 * ((rows - 100) / 6.) ** 2) + np.exp(-0.5 * ((rows - 160) / 6.) ** 2))
    img[80:180, 120:123] = prof[:, None] + np.array(clip, np.float32)[None, :]
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
        self.assertEqual((nsat > 0).sum(), 2)  # stock labelling splits the bloom in two
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


class TestProminence(unittest.TestCase):

    def glow_frame(self):
        # saturated core on a wide glow carrying two small bumps
        img = flat_frame(7)
        yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
        r = np.hypot(yy - 128., xx - 128.)
        img += scipy.ndimage.gaussian_filter(200 * (r < 50).astype(np.float32), 3)   # flat glow
        img += 3000 * np.exp(-r ** 2 / (2 * 8. ** 2))
        for y, x in ((128, 165), (90, 128)):
            img += 20 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 2. ** 2))   # rise 20 < threshold 25
        img[124:133, 124:133] = 60000
        return img

    def test_glow_bumps_join_the_spot(self):
        img = self.glow_frame()
        self.assertGreater(len(make_worker(SAT).peaksearch(img.copy())), 1)
        pks = make_worker(SAT, prominence=True).peaksearch(img.copy())
        self.assertEqual(len(pks), 1)
        self.assertEqual(pks[0, cid.NPROPERTY], 81)

    def test_saturated_bloom_one_peak(self):
        # pile-up level differs between the bloom columns: still one peak
        img = bloom_frame(clip=(0, -1500, 1000))
        pks = make_worker(SAT, prominence=True).peaksearch(img.copy())
        self.assertEqual((pks[:, cid.NPROPERTY] > 0).sum(), 1)
        self.assertEqual(pks[:, cid.NPROPERTY].sum(), (img >= SAT).sum())

    def test_clipped_column_inside_bloom(self):
        # a 1 px column inside the bloom written at a fixed lower value (43673)
        img = bloom_frame()
        img[80:180, 121] = 43673
        pks = make_worker(SAT, prominence=True, clipped=[43673]).peaksearch(img.copy())
        self.assertEqual((pks[:, cid.NPROPERTY] > 0).sum(), 1)
        self.assertEqual(pks[:, cid.NPROPERTY].sum(), ((img >= SAT) | (img == 43673)).sum())

    def test_touching_spots_stay_apart(self):
        img = flat_frame(8)
        yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
        for x in (120, 132):
            img += 400 * np.exp(-((yy - 128.) ** 2 + (xx - x) ** 2) / (2 * 3. ** 2))
        self.assertEqual(len(make_worker(None, prominence=True).peaksearch(img)), 2)

    def test_isolated_spots_as_before(self):
        img = flat_frame(9)
        for y, x, a in ((50, 50, 60), (150, 60, 300), (60, 180, 2000)):
            img[y - 1:y + 2, x - 1:x + 2] += a
        a = make_worker(None).peaksearch(img.copy())
        b = make_worker(None, prominence=True).peaksearch(img.copy())
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.allclose(a[:, cid.s_raw], b[:, cid.s_raw]))


class TestNoScatter(unittest.TestCase):

    def test_static_background_subtracted_unscaled(self):
        # background = dark only (no scatter): per-pixel structure, frame offset drifts by 3 ADU
        rs = np.random.RandomState(1)
        bg = (94 + 20 * (np.arange(SHAPE[1]) >= 128)[None, :] + np.zeros(SHAPE)).astype(np.float32)
        img = bg + 3 + rs.normal(0, 4, SHAPE).astype(np.float32)
        img[100:104, 60:64] += 400
        w = worker(bg.copy(), np.zeros(SHAPE, np.uint8), threshold=25, m_ratio_thresh=None)
        cor = w.bgsub(img.copy())
        self.assertAlmostEqual(w.offset, 3, delta=0.2)
        self.assertLess(abs(np.median(cor)), 0.2)
        self.assertAlmostEqual(cor[100:104, 60:64].mean(), 400, delta=5)
        pks = w.peaksearch(img.copy())
        self.assertEqual(len(pks), 1)


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
