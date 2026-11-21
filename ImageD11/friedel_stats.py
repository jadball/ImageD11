"""
Friedel pairs for box-beam 3DXRD, with the matching windows and cuts measured
from the data.

A grain diffracts g and -g: two spots, usually near omega and omega + 180. With
g_s(t) the sample-frame g-vector of a spot computed for a grain at translation
t, a pair satisfies g_s1(t) + g_s2(t) = 0 at the grain's t. So

- a pair gives a g-vector, (g_s1 - g_s2) / 2, without knowing t: one pair fixes
  t only along a line, and g barely changes along it (not at all when wedge =
  chi = 0, as friedel_pairs.update_geometry_fpairs, Jacob et al. 2024);
- the pairs of one grain fix its t (least squares on g_s1 + g_s2).

- predict_partners : where -g of each peak diffracts (with wedge and chi)
- pair_peaks : candidate partners near where -g diffracts, scored by how
  much more often real pairs than chance ones (partners searched at a wrong
  omega) have their omega mismatch, grain position and intensity; cut where
  pairs stop outnumbering chance; pairs_within for another cut
- partner_observable, partner_fraction : which peaks could have a partner
  (detector, mask), and down to which intensity peaks pair more than chance
- pair_colfile : the paired peaks with the pair g-vectors
- ring_windows : ring centre, width and d* window from the paired peaks
- LineIndexer : orientations from two pairs whose g-vectors make an allowed
  angle and whose position lines cross; scored on the pairs that index and
  pass through that position; minpks from scrambled peaks (each g turned
  about the rotation axis by a random angle)
- fit_positions : grain positions from their pairs
"""

from __future__ import print_function, division

import numba
import numpy as np
import scipy.ndimage
import scipy.spatial
from scipy.special import erfinv
from ImageD11 import transform

# (omega, eta, tth, log10 I); omega and eta are periodic
_PERIODIC = np.array([True, True, False, False])
NAMES = ('omega', 'eta', 'tth', 'logI')
# per candidate pair (pair_peaks)
FEATURES = ('rho', 'axis distance', 'height', 'dlogI')


# ------------------------------------------------------------------ geometry

def _axis_matrix(pars):
    """ wedge . chi, as in cImageD11.compute_gv (its matmat(cmat, wmat) is wmat . cmat) """
    w, x = np.radians(float(pars['wedge'])), np.radians(float(pars['chi']))
    W = np.array([[np.cos(w), 0, -np.sin(w)], [0, 1, 0], [np.sin(w), 0, np.cos(w)]])
    C = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    return W.dot(C)


def gv_rows(xyz, omega, t, pars):
    """ g-vectors as cImageD11.compute_gv, with a translation per row
    (t is (3,) or (n, 3)) """
    mat = _axis_matrix(pars)
    om = np.radians(np.asarray(omega) * float(pars['omegasign']))
    c, s = np.cos(om), np.sin(om)
    t = np.broadcast_to(np.asarray(t, float), xyz.shape)
    u = np.column_stack([c * t[:, 0] - s * t[:, 1], s * t[:, 0] + c * t[:, 1], t[:, 2]])
    d = xyz - u.dot(mat.T)
    d /= np.sqrt((d * d).sum(axis=1))[:, None]
    k = (d - np.array([1.0, 0, 0])) / float(pars['wavelength'])
    v = k.dot(mat)
    return np.column_stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]])


def lab_xyz(cf):
    return np.ascontiguousarray(transform.compute_xyz_lab([cf.sc, cf.fc], **cf.parameters.parameters).T)


def _wrap180(x):
    return (x + 180) % 360 - 180


def predict_partners(cf):
    """
    Where -g of each peak (g for a grain at the origin) diffracts: omega,
    eta, tth of the solution nearest omega + 180. valid is False where -g
    cannot diffract.
    """
    pars = cf.parameters.parameters
    wvln, wedge, chi = float(pars['wavelength']), float(pars['wedge']), float(pars['chi'])
    g = gv_rows(lab_xyz(cf), cf.omega, np.zeros(3), pars)
    # omega convention of uncompute_g_vectors: the one that gives back the peaks themselves
    _, _, oms = transform.uncompute_g_vectors(g.T, wvln, wedge=wedge, chi=chi)
    sign = None
    for sg in (1.0, -1.0):
        if np.median(np.min([np.abs(_wrap180(o * sg - cf.omega)) for o in oms], axis=0)) < 1e-6:
            sign = sg
            break
    if sign is None:
        print('predict_partners: uncompute_g_vectors does not give back the peaks; check wedge, chi, omegasign')
        sign = 1.0
    tth, etas, oms = transform.uncompute_g_vectors(-g.T, wvln, wedge=wedge, chi=chi)
    oms = [o * sign for o in oms]
    k = np.argmin([np.abs(_wrap180(o - cf.omega - 180)) for o in oms], axis=0)
    om = np.choose(k, oms)
    eta = np.choose(k, etas)
    return om, eta, tth, tth > 0


# ------------------------------------------------------------------ pairing

def _space(cf, partner=None, domega=0.0):
    if partner is None:
        om, eta, tth = cf.omega, cf.eta, cf.tth
    else:   # where the partner is expected
        om, eta, tth = partner[0] + domega, partner[1], partner[2]
    return np.column_stack([om % 360, eta % 360, tth, np.log10(np.maximum(cf.sum_intensity, 1.0))])


def _nearest(X, Q, scale):
    """ nearest point of X to each point of Q, distance in units of scale,
    periodic in omega and eta """
    lo = np.where(_PERIODIC, 0, np.minimum(X.min(axis=0), Q.min(axis=0)))
    hi = np.where(_PERIODIC, 360, np.maximum(X.max(axis=0), Q.max(axis=0)))
    box = np.where(_PERIODIC, 360 / scale, 2 * (hi - lo) / scale + 1)
    tree = scipy.spatial.cKDTree(np.mod((X - lo) / scale, box), boxsize=box)
    return tree.query(np.mod((Q - lo) / scale, box), k=1)


def _wrap(x):
    x = x.copy()
    x[:, _PERIODIC] = _wrap180(x[:, _PERIODIC])
    return x


def _nearest_pairs(cf, pred, null_offsets, niter):
    """ mutual nearest partners in (omega, eta, tth, log I), distances in
    spreads re-measured on the pairs, cut where N(d < r) - 2 N_null(d < r) is
    largest (see pair_peaks) """
    X = _space(cf)
    n = len(X)
    valid = pred[3]
    Q0 = _space(cf, pred)
    scale = np.array([1.0, 1.0, 1.0, 1e9])   # first pass: angles only
    centre = np.zeros(4)
    cut = np.inf
    for _ in range(niter):
        d, j = _nearest(X, Q0 + centre, scale)
        d[~valid] = np.inf
        sel = (j[j] == np.arange(n)) & (d < cut)
        diff = _wrap(X[j[sel]] - Q0[sel])
        centre = np.median(diff, axis=0)
        scale = 1.4826 * np.median(np.abs(diff - centre), axis=0)
        d, j = _nearest(X, Q0 + centre, scale)
        d[~valid] = np.inf
        dn = np.array([_nearest(X, _space(cf, pred, o) + centre, scale)[0] for o in null_offsets])
        dn[:, ~valid] = np.inf
        r = np.linspace(0, np.percentile(d[valid], 99), 500)
        n_obs = np.searchsorted(np.sort(d), r)
        n_null = np.searchsorted(np.sort(dn.ravel()), r) / len(null_offsets)
        cut = r[np.argmax(n_obs - 2 * n_null)]
    return scale, centre


def _window(cf, pred, scale, centre, null_offsets):
    """ angular search radius (spreads): where the nearest peak to the
    predicted partner stops being more often real than chance, the r that
    maximises N(d < r) - N_null(d < r) """
    sc = scale.copy()
    sc[3] = 1e9
    X = _space(cf)
    valid = pred[3]
    d = _nearest(X, _space(cf, pred) + centre, sc)[0][valid]
    dn = np.concatenate([_nearest(X, _space(cf, pred, o) + centre, sc)[0][valid] for o in null_offsets])
    r = np.linspace(0, np.percentile(d, 99), 500)
    gain = np.searchsorted(np.sort(d), r) - np.searchsorted(np.sort(dn), r) / len(null_offsets)
    return r[np.argmax(gain)]


def _candidates(cf, pred, scale, centre, window, domega):
    """ all (peak, candidate partner) within window spreads in (omega, eta,
    tth) of where each peak's partner is predicted, shifted by domega """
    sc = scale[:3]
    box = np.array([360 / sc[0], 360 / sc[1], 2 * np.ptp(cf.tth) / sc[2] + 1])
    lo = np.array([0, 0, cf.tth.min() - np.ptp(cf.tth)])
    X = np.mod((_space(cf)[:, :3] - lo) / sc, box)
    Q = np.mod((_space(cf, pred, domega)[:, :3] + centre[:3] - lo) / sc, box)
    vi = np.flatnonzero(pred[3])
    lists = scipy.spatial.cKDTree(X, boxsize=box).query_ball_point(Q[vi], window)
    a = np.repeat(vi, [len(l) for l in lists])
    b = np.fromiter((k for l in lists for k in l), dtype=int, count=len(a))
    return a, b


def _pair_solve(x1, o1, x2, o2, pars, niter=6, h=100.0):
    """ g_s1(t) + g_s2(t) = 0 per pair: Gauss-Newton for the t nearest the
    origin along the two directions one pair fixes (h: finite difference step
    in t). Returns t, g_s1(t), g_s2(t), the unit vector of the direction the
    residual keeps whatever t, and the direction of the line of t that
    satisfy the pair as well (t + s v). """
    t = np.zeros((len(x1), 3))
    for _ in range(niter):
        r = _pair_residual(x1, o1, x2, o2, t, pars)
        J = np.empty((len(x1), 3, 3))
        for k in range(3):
            dt = np.zeros(3)
            dt[k] = h
            J[:, :, k] = (_pair_residual(x1, o1, x2, o2, t + dt, pars) -
                          _pair_residual(x1, o1, x2, o2, t - dt, pars)) / (2 * h)
        U, S, Vt = np.linalg.svd(J)
        c = np.einsum('nji,nj->ni', U[:, :, :2], r) / S[:, :2]
        t -= np.einsum('nij,ni->nj', Vt[:, :2, :], c)
    return t, gv_rows(x1, o1, t, pars), gv_rows(x2, o2, t, pars), U[:, :, 2], Vt[:, 2, :]


def _pair_features(cf, xyz, a, b, domega, chunk=200000):
    """ per candidate pair: rho, the part of g_s1 + g_s2 (over |g|) that no
    grain position removes; the grain position nearest the axis, as distance
    from the axis and height (um); difference of log10 intensity """
    pars = cf.parameters.parameters
    F = np.empty((len(a), 4))
    logI = np.log10(np.maximum(cf.sum_intensity, 1.0))
    for s in range(0, len(a), chunk):
        i, j = a[s:s + chunk], b[s:s + chunk]
        t, g1, g2, u, _ = _pair_solve(xyz[i], cf.omega[i], xyz[j], cf.omega[j] - domega, pars)
        gn = np.sqrt((((g1 - g2) / 2) ** 2).sum(axis=1))
        F[s:s + chunk, 0] = ((g1 + g2) * u).sum(axis=1) / gn
        F[s:s + chunk, 1] = np.hypot(t[:, 0], t[:, 1])
        F[s:s + chunk, 2] = t[:, 2]
        F[s:s + chunk, 3] = logI[j] - logI[i]
    return F


def _best(n, a, b, score):
    """ highest scoring candidate of each peak: score (-inf if none), partner
    (the peak itself if none) """
    o = np.lexsort((-score, a))
    first = np.r_[True, a[o][1:] != a[o][:-1]]
    bs = np.full(n, -np.inf)
    bj = np.arange(n)
    bs[a[o][first]] = score[o][first]
    bj[a[o][first]] = b[o][first]
    return bs, bj


def pair_peaks(cf, null_offsets=(7.1, -7.1, 13.3, -13.3), niter=4, nbins=60):
    """
    Friedel pairs in a columnfile (omega, eta, tth, sum_intensity, sc, fc).

    1. Each peak's partner is predicted where its -g diffracts
       (predict_partners). Mutual nearest partners in (omega, eta, 2theta,
       log I) give the spread of each coordinate (1.4826 MAD about the median
       offset). The search radius is where the nearest peak stops being more
       often a real partner than a chance one.
    2. Every peak within that radius is a candidate. For each candidate pair:
       rho, the part of the pair equation g_s1(t) + g_s2(t) = 0 that no grain
       position t removes (mostly the omega mismatch); the grain position
       nearest the rotation axis (distance from the axis, height); the
       difference in log I.
    3. Chance candidates: the same search null_offsets degrees away in omega
       (far beyond the omega spread of real pairs), moved back by the offset
       so they are chance peaks at the predicted place.
    4. Score: the sum over features of log(density of real / density of
       chance), from histograms of the candidates (real = found minus chance)
       in bins that each hold the same number of chance candidates. Each peak
       keeps its best candidate. The cut maximises N - 2 N_null: beyond it a
       lower score adds more chance partners than real ones.

    Returns a dict: i1, i2 (pairs: mutual best, within the cut), cut, and
    r, n_obs, n_null (the curves); d and j (minus the score of each peak's
    best candidate, and that candidate: pairs_within selects pairs at another
    cut), d_null (the same for chance, per offset); scale, centre (spreads and
    median offset of step 1), window (search radius, spreads); features (name,
    bin edges, candidate counts found and chance per bin, log ratio);
    partner (predict_partners).
    """
    n = cf.nrows
    pred = predict_partners(cf)
    scale, centre = _nearest_pairs(cf, pred, null_offsets, niter)
    window = _window(cf, pred, scale, centre, null_offsets)
    xyz = lab_xyz(cf)
    cand = []
    for o in (0.0,) + tuple(null_offsets):
        a, b = _candidates(cf, pred, scale, centre, window, o)
        cand.append((a, b, _pair_features(cf, xyz, a, b, o)))
    F0 = cand[0][2]
    Fn = np.concatenate([c[2] for c in cand[1:]])
    m = float(len(null_offsets))
    lr = np.zeros(len(F0)), [np.zeros(len(c[0])) for c in cand[1:]]
    features = []
    for k, name in enumerate(FEATURES):
        edges = np.unique(np.percentile(Fn[:, k], np.linspace(0, 100, nbins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        h0 = np.histogram(F0[:, k], edges)[0].astype(float)
        hn = np.histogram(Fn[:, k], edges)[0] / m
        real = np.maximum(h0 - hn, 0) + 1
        ratio = np.log((real / real.sum()) / ((hn + 1) / (hn + 1).sum()))
        features.append((name, edges, h0, hn, ratio))
        nb = len(ratio)
        lr[0][:] += ratio[np.clip(np.searchsorted(edges, F0[:, k]) - 1, 0, nb - 1)]
        for c, l in zip(cand[1:], lr[1]):
            l += ratio[np.clip(np.searchsorted(edges, c[2][:, k]) - 1, 0, nb - 1)]
    bs, j = _best(n, cand[0][0], cand[0][1], lr[0])
    d = -bs
    dn = np.array([-_best(n, c[0], c[1], l)[0] for c, l in zip(cand[1:], lr[1])])
    fin = d[np.isfinite(d)]
    r = np.linspace(fin.min(), np.percentile(fin, 99), 500)
    n_obs = np.searchsorted(np.sort(d), r)
    n_null = np.searchsorted(np.sort(dn.ravel()), r) / m
    cut = r[np.argmax(n_obs - 2 * n_null)]
    P = {'cut': cut, 'scale': scale, 'centre': centre, 'window': window, 'r': r, 'n_obs': n_obs,
         'n_null': n_null, 'd': d, 'j': j, 'd_null': dn, 'partner': pred, 'features': features}
    P['i1'], P['i2'] = pairs_within(P, cut)
    return P


def pairs_within(P, cut):
    """ pairs (i1, i2) of pair_peaks result P that are mutual nearest partners
    closer than cut """
    d, j = P['d'], P['j']
    i = np.arange(len(d))
    keep = (j[j] == i) & (d < cut) & (i < j)
    return i[keep], j[keep]


def partner_observable(cf, P, mask=None, keep=0.95):
    """
    Whether each peak's partner could be seen: where -g diffracts
    (P['partner'], for a grain at the origin) is on the detector, and no
    masked pixel (mask: 1 = masked, raw pixels) is within the distance a real
    partner lies from that prediction (the fraction keep of the pairs found).
    Without a mask, the detector is taken as the area the peaks cover.
    Returns a bool array and that distance in pixels.
    """
    pars = cf.parameters.parameters
    om, eta, tth, valid = P['partner']
    fc, sc = transform.compute_xyz_from_tth_eta(tth, eta, om, **pars)
    # corrected -> raw pixels: the distortion at the nearest peak
    tree = scipy.spatial.cKDTree(np.column_stack([cf.sc, cf.fc]))
    _, k = tree.query(np.column_stack([sc, fc]))
    s_raw = sc + (cf.s_raw - cf.sc)[k]
    f_raw = fc + (cf.f_raw - cf.fc)[k]
    i1, i2 = P['i1'], P['i2']
    miss = np.concatenate([np.hypot(s_raw[i1] - cf.s_raw[i2], f_raw[i1] - cf.f_raw[i2]),
                           np.hypot(s_raw[i2] - cf.s_raw[i1], f_raw[i2] - cf.f_raw[i1])])
    margin = np.percentile(miss, 100 * keep)
    if mask is None:
        lo = np.array([cf.s_raw.min(), cf.f_raw.min()])
        hi = np.array([cf.s_raw.max(), cf.f_raw.max()])
        inside = ((s_raw - margin > lo[0]) & (s_raw + margin < hi[0]) &
                  (f_raw - margin > lo[1]) & (f_raw + margin < hi[1]))
        return valid & inside, margin
    clear = scipy.ndimage.distance_transform_edt(np.asarray(mask) == 0)   # distance to a masked pixel
    si, fi = np.round(s_raw).astype(int), np.round(f_raw).astype(int)
    on = (si >= 0) & (si < clear.shape[0]) & (fi >= 0) & (fi < clear.shape[1]) & valid
    ok = np.zeros(len(si), bool)
    ok[on] = (clear[si[on], fi[on]] > margin) & \
        (si[on] > margin) & (si[on] < clear.shape[0] - 1 - margin) & \
        (fi[on] > margin) & (fi[on] < clear.shape[1] - 1 - margin)
    return ok, margin


def partner_fraction(P, logI, observable, nbins=40, cut=None):
    """
    Among peaks whose partner could be seen, in equal-count bins of log10
    intensity: the fraction with a partner inside the cut and the fraction a
    chance search finds (mean over the null offsets). The floor: going down
    from the bin with the most real partners, the first intensity below which
    a bin has no more real partners than chance ones (fraction <= 2 x chance).
    Returns centres, edges, fraction, chance fraction, floor (log10 I).
    """
    cut = P['cut'] if cut is None else cut
    x = logI[observable]
    edges = np.unique(np.percentile(x, np.linspace(0, 100, nbins + 1)))
    k = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    n = np.bincount(k, minlength=len(edges) - 1).astype(float)
    f = np.bincount(k, P['d'][observable] < cut, len(edges) - 1) / np.maximum(n, 1)
    fn = np.mean([np.bincount(k, dn[observable] < cut, len(edges) - 1) for dn in P['d_null']], axis=0) / np.maximum(n, 1)
    best = np.argmax(f - 2 * fn)
    bad = np.flatnonzero(f[:best] <= 2 * fn[:best])
    floor = edges[bad[-1] + 1] if len(bad) else edges[0]
    return 0.5 * (edges[1:] + edges[:-1]), edges, f, fn, floor


# ------------------------------------------------------------ pair g-vectors

def _pair_residual(xyz1, om1, xyz2, om2, t, pars):
    return gv_rows(xyz1, om1, t, pars) + gv_rows(xyz2, om2, t, pars)


def pair_g(cf, i1, i2, niter=6, h=100.0):
    """
    Position-free g-vector of each pair: (g_s1 - g_s2) / 2 at the t nearest
    the origin with g_s1(t) + g_s2(t) = 0 (Gauss-Newton on the two directions
    one pair determines; h is the finite difference step in t).
    Returns g (n, 3) and |g_s1 + g_s2| / |g| left at the end.
    """
    pars = cf.parameters.parameters
    xyz = lab_xyz(cf)
    _, g1, g2, _, _ = _pair_solve(xyz[i1], cf.omega[i1], xyz[i2], cf.omega[i2], pars, niter, h)
    g = (g1 - g2) / 2
    gn = np.sqrt((g * g).sum(axis=1))
    return g, np.sqrt(((g1 + g2) ** 2).sum(axis=1)) / gn


def pair_colfile(cf, i1, i2):
    """ The peaks of pairs (i1, i2) with the pair g-vectors (gx, gy, gz, ds,
    tth; -g for the second peak). Columns 'pair' (pair number) and 'peak' (row
    in cf) are added. """
    g, _ = pair_g(cf, i1, i2)
    rows = np.concatenate((i1, i2))
    cfp = cf.copyrows(rows)
    gg = np.concatenate((g, -g))
    ds = np.sqrt((gg * gg).sum(axis=1))
    wvln = float(cf.parameters.get('wavelength'))
    for name, col in zip(('gx', 'gy', 'gz', 'ds'), (gg[:, 0], gg[:, 1], gg[:, 2], ds)):
        cfp.addcolumn(col, name)
    cfp.addcolumn(np.degrees(2 * np.arcsin(ds * wvln / 2)), 'tth')
    cfp.addcolumn(np.concatenate((np.arange(len(i1)), np.arange(len(i1)))), 'pair')
    cfp.addcolumn(rows, 'peak')
    return cfp


# ------------------------------------------------------------------- rings

def _gauss_flat(x, w, niter=200):
    """ Gaussian plus a flat background on |x| < w (EM): centre, sigma,
    background fraction """
    x = x[np.abs(x) < w]
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    m = np.median(x)
    s = max(1.4826 * np.median(np.abs(x - m)), 1e-9)
    f = 0.1
    for _ in range(niter):
        g = (1 - f) * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        p = g / (g + f / (2 * w))
        f = min(max(1 - p.mean(), 1e-6), 1 - 1e-6)
        m = (p * x).sum() / p.sum()
        s = np.sqrt((p * (x - m) ** 2).sum() / p.sum())
    return m, s, f


def ring_windows(ds, ringds, rings):
    """
    For each ring in rings, from the d* of paired peaks (no grain position
    broadening): centre offset, width (Gaussian plus a flat background over
    half the gap to the next rings) and a window: the offset plus the
    distance at which the ring's density falls to the background (at most
    half the gap).
    Returns a dict ring -> (offset, sigma, background fraction, window).
    """
    r = np.asarray(ringds, float)
    k = np.clip(np.searchsorted(r, ds), 1, len(r) - 1)
    k = np.where(np.abs(ds - r[k - 1]) < np.abs(ds - r[k]), k - 1, k)
    out = {}
    for ring in rings:
        w = np.min(np.abs(np.delete(r, ring) - r[ring])) / 2
        m, s, f = _gauss_flat(ds[k == ring] - r[ring], w)
        if not np.isfinite(s):
            continue
        # (1 - f) N(x; 0, s) = f / (2 w)
        q = (1 - f) * 2 * w / (f * s * np.sqrt(2 * np.pi))
        x = s * np.sqrt(2 * np.log(q)) if q > 1 else 0.0
        out[ring] = (m, s, f, min(abs(m) + x, w))
    return out


# ---------------------------------------------------------------- indexing

def _ring_numbers(ringds, dstars):
    r = np.asarray(ringds)
    return [int(np.argmin(np.abs(r - d))) for d in dstars]


def pair_lines(cf, i1, i2):
    """ The line of grain positions t that each pair (i1, i2) allows: the
    point nearest the rotation axis and the unit direction. """
    xyz = lab_xyz(cf)
    t, _, _, _, v = _pair_solve(xyz[i1], cf.omega[i1], xyz[i2], cf.omega[i2], cf.parameters.parameters)
    return t, v


def _turn_z(g, rng):
    a = rng.uniform(0, 2 * np.pi, len(g))
    c, s = np.cos(a), np.sin(a)
    return np.column_stack([c * g[:, 0] - s * g[:, 1], s * g[:, 0] + c * g[:, 1], g[:, 2]])


@numba.njit(cache=True)
def _trial_row(i, n, X, I2, coses, ctol, dmax, smin, fill, A, B, D, S, k):
    """ trials (i, j), j in I2: n[i].n[j] within ctol of one of coses, the
    lines X[i], X[j] (t0, v; same rows as n) within dmax and the sine of their
    angle above smin; as _crossing. Counts them, and stores them from k if fill """
    m = 0
    for j in I2:
        if j == i:
            continue
        c = n[i, 0] * n[j, 0] + n[i, 1] * n[j, 1] + n[i, 2] * n[j, 2]
        hit = False
        for x in coses:
            if abs(c - x) < ctol:
                hit = True
                break
        if not hit:
            continue
        w0 = X[i, 0] - X[j, 0]
        w1 = X[i, 1] - X[j, 1]
        w2 = X[i, 2] - X[j, 2]
        bb = X[i, 3] * X[j, 3] + X[i, 4] * X[j, 4] + X[i, 5] * X[j, 5]
        d1 = X[i, 3] * w0 + X[i, 4] * w1 + X[i, 5] * w2
        d2 = X[j, 3] * w0 + X[j, 4] * w1 + X[j, 5] * w2
        den = max(1 - bb * bb, 1e-12)
        sn = np.sqrt(den)
        if sn <= smin:
            continue
        s = (bb * d2 - d1) / den
        u = (d2 - bb * d1) / den
        q0 = (X[i, 0] + s * X[i, 3]) - (X[j, 0] + u * X[j, 3])
        q1 = (X[i, 1] + s * X[i, 4]) - (X[j, 1] + u * X[j, 4])
        q2 = (X[i, 2] + s * X[i, 5]) - (X[j, 2] + u * X[j, 5])
        d = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2)
        if d >= dmax:
            continue
        if fill:
            A[k + m] = i
            B[k + m] = j
            D[k + m] = d
            S[k + m] = sn
        m += 1
    return m


@numba.njit(parallel=True, cache=True)
def _trial_count(n, X, I1, I2, coses, ctol, dmax, smin, counts):
    e = np.zeros(0, np.int64)
    f = np.zeros(0)
    for r in numba.prange(len(I1)):
        counts[r] = _trial_row(I1[r], n, X, I2, coses, ctol, dmax, smin, False, e, e, f, f, 0)


@numba.njit(parallel=True, cache=True)
def _trial_fill(n, X, I1, I2, coses, ctol, dmax, smin, start, A, B, D, S):
    for r in numba.prange(len(I1)):
        _trial_row(I1[r], n, X, I2, coses, ctol, dmax, smin, True, A, B, D, S, start[r])


def _line_trials(n, X, I1, I2, coses, ctol, dmax, smin):
    """ rows (a in I1, b in I2) of unit g-vectors n whose angle matches one of
    coses within ctol and whose lines X (t0, v per row) cross within dmax at a
    sine above smin; returns a, b, distance, sine """
    I1 = np.ascontiguousarray(I1, np.int64)
    I2 = np.ascontiguousarray(I2, np.int64)
    coses = np.asarray(coses, float)
    counts = np.zeros(len(I1), np.int64)
    _trial_count(n, X, I1, I2, coses, ctol, dmax, smin, counts)
    start = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    N = int(counts.sum())
    A, B = np.empty(N, np.int64), np.empty(N, np.int64)
    D, S = np.empty(N), np.empty(N)
    _trial_fill(n, X, I1, I2, coses, ctol, dmax, smin, start, A, B, D, S)
    return A, B, D, S


def _crossing(p1, v1, p2, v2):
    """ closest approach of lines p + s v: distance, sine of the angle between
    them and the midpoint """
    w = p1 - p2
    b = (v1 * v2).sum(axis=1)
    d1, d2 = (v1 * w).sum(axis=1), (v2 * w).sum(axis=1)
    den = np.maximum(1 - b * b, 1e-12)
    s, u = (b * d2 - d1) / den, (d2 - b * d1) / den
    q1, q2 = p1 + s[:, None] * v1, p2 + u[:, None] * v2
    return np.sqrt(((q1 - q2) ** 2).sum(axis=1)), np.sqrt(den), 0.5 * (q1 + q2)


def _half_normal_on(d, dn, w, niter=300, nfine=2000):
    """ d on [0, w) as (1 - f) half-normal(sigma) + f (density of dn): sigma.
    Fitted on a histogram of d with nfine bins. """
    edges = np.linspace(0, w, 81)
    hb = np.histogram(dn[dn < w], edges, density=True)[0]
    fine = np.linspace(0, w, nfine + 1)
    cnt = np.histogram(d[d < w], fine)[0].astype(float)
    x = 0.5 * (fine[1:] + fine[:-1])
    x, cnt = x[cnt > 0], cnt[cnt > 0]
    bg = hb[np.clip(np.searchsorted(edges, x) - 1, 0, len(hb) - 1)]
    s, f = w / 10.0, 0.5
    for _ in range(niter):
        g = (1 - f) * 2 * np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))
        p = cnt * g / (g + f * bg)
        f = 1 - p.sum() / cnt.sum()
        s = np.sqrt((p * x * x).sum() / p.sum())
    return s


class LineIndexer(object):
    """
    Index Friedel pairs using where their grains can be: each pair allows a
    line of grain positions (pair_lines). Two pairs from one grain have
    lines that cross at the grain.

    For a pair of rings, every two pairs whose g-vectors make an allowed angle
    (unitcell.getcoses, within cosine_tol) and whose lines pass within line_tol
    give a trial: an orientation (unitcell.orient) and a grain position (where
    the lines cross). A trial's score is the number of pairs that index
    (within hkl_tol) and whose line also passes within line_tol of that
    position. Trials are taken best first; each kept one claims its pairs.
    minpks: the best score of as many trials made from scrambled g-vectors
    (each turned about the rotation axis by a random angle; lines unchanged).

    line_tol (measure_line_tol): the spread of the distance between the lines
    of two pairs of one grain (half-normal on top of the scrambled distances),
    and the distance that keeps the fraction keep of them. It can be set by
    hand before run.
    """

    def __init__(self, cfp, t, v, ucell, score_ds, dstol, cosine_tol, keep=0.95, seed=0):
        self.ucell = ucell
        self.dstol = dstol
        self.cosine_tol = cosine_tol
        self.keep = keep
        self.pair = cfp['pair'].astype(int)
        self.t, self.v = t, v
        self._X = np.ascontiguousarray(np.hstack((t, v)), float)
        self._Xrow = self._X[self.pair]   # the line of each row
        g = np.column_stack([cfp.gx, cfp.gy, cfp.gz])
        ringds = np.asarray(ucell.ringds)
        self.ra = np.argmin(np.abs(cfp.ds[:, None] - ringds[None, :]), axis=1)
        self.ra[np.abs(cfp.ds - ringds[self.ra]) >= dstol] = -1
        self.S = np.flatnonzero(np.isin(self.ra, _ring_numbers(ringds, score_ds)))
        self.g = {False: np.ascontiguousarray(g), True: np.ascontiguousarray(_turn_z(g, np.random.RandomState(seed)))}
        self.n = {k: np.ascontiguousarray(x / np.sqrt((x * x).sum(axis=1))[:, None]) for k, x in self.g.items()}
        self._cache = {}
        self.first = np.arange(len(g)) < len(g) // 2   # one row per pair (the other is -g)
        self.claimed = np.zeros(len(t), bool)
        self.rng = np.random.RandomState(seed)
        self.line_tol = None
        self.ubis, self.positions, self.npairs = [], [], []
        self.pair_grain = np.full(len(t), -1)
        r = np.sqrt((t[:, :2] ** 2).sum(axis=1))
        self.radius = np.percentile(r, 100 * keep)

    def _trials(self, r1, r2, scrambled, dmax=None, smin=None):
        """ trials of rings r1, r2 (one row per pair, pairs unclaimed):
        a, b, distance and sine of the crossing of their lines, crossing point.
        Default dmax, smin: line_tol, line_tol / radius (cached, as claims only
        remove trials) """
        if dmax is None:
            dmax, smin = self.line_tol, self.line_tol / self.radius
        key = (r1, r2, scrambled, dmax, smin)
        if key not in self._cache:
            coses = self.ucell.getcoses(r1, r2)
            free = self.first & ~self.claimed[self.pair]
            I1 = np.flatnonzero(free & (self.ra == r1))
            I2 = np.flatnonzero(free & (self.ra == r2))
            if len(coses) == 0 or len(I1) == 0 or len(I2) == 0:
                self._cache[key] = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
            else:
                self._cache[key] = _line_trials(self.n[scrambled], self._Xrow, I1, I2, coses, self.cosine_tol, dmax, smin)
        a, b, d, sin = self._cache[key]
        k = ~(self.claimed[self.pair[a]] | self.claimed[self.pair[b]])
        a, b, d, sin = a[k], b[k], d[k], sin[k]
        pa, pb = self.pair[a], self.pair[b]
        T = _crossing(self.t[pa], self.v[pa], self.t[pb], self.v[pb])[2]
        return a, b, d, sin, T

    def measure_line_tol(self, ring_pairs):
        """ line_tol from the trials of all ring_pairs ((r1, r2), ...);
        returns the spread (sigma) and line_tol """
        w = 2 * self.radius
        d, dn = [], []
        free = self.first & ~self.claimed[self.pair]
        for r1, r2 in ring_pairs:
            coses = self.ucell.getcoses(r1, r2)
            I1 = np.flatnonzero(free & (self.ra == r1))
            I2 = np.flatnonzero(free & (self.ra == r2))
            if len(coses) and len(I1) and len(I2):
                for x, scrambled in ((d, False), (dn, True)):
                    x.append(_line_trials(self.n[scrambled], self._Xrow, I1, I2, coses, self.cosine_tol, w, -1.0)[2])
        d, dn = np.concatenate(d), np.concatenate(dn)
        sigma = _half_normal_on(d, dn, w)
        self.line_d, self.line_dn = d, dn
        # half-normal: the fraction keep lies within sigma * sqrt(2) * erfinv(keep)
        self.line_tol = sigma * np.sqrt(2) * erfinv(self.keep)
        self.line_sigma = sigma
        return sigma, self.line_tol

    def _ok(self):
        """ per pair: on a scoring ring and not claimed """
        ok = np.zeros(len(self.t), bool)
        ok[self.pair[self.S]] = True
        return ok & ~self.claimed

    def _orient(self, scrambled, a, b, r1, r2):
        """ unitcell.orient for many trials at once (the hkl pair whose
        angle is nearest, as orient with crange < 0) """
        g = self.g[scrambled]
        hab, c2ab, matrs = self.ucell.getanglehkls(r1, r2)
        c2ab = np.asarray(c2ab)
        g1, g2 = g[a], g[b]
        cos = (g1 * g2).sum(axis=1) / np.sqrt((g1 * g1).sum(axis=1) * (g2 * g2).sum(axis=1))
        i = np.searchsorted(c2ab, cos, side='left')
        lo = np.clip(i - 1, 0, len(c2ab) - 1)
        hi = np.clip(i, 0, len(c2ab) - 1)
        best = np.where((i > 0) & ((i == len(c2ab)) | (np.abs(cos - c2ab[lo]) < np.abs(cos - c2ab[hi]))), lo, hi)
        u0 = g1 / np.sqrt((g1 * g1).sum(axis=1))[:, None]
        u2 = np.cross(g1, g2)
        u2 /= np.sqrt((u2 * u2).sum(axis=1))[:, None]
        M = np.stack((u0, np.cross(u0, u2), u2), axis=1)
        return np.ascontiguousarray(np.matmul(np.asarray(matrs)[best], M))

    def _index(self):
        """ lines sorted for _line_hits (rebuilt when line_tol changes): bins
        narrow enough that a point within radius moves by line_tol / 4 """
        if getattr(self, '_index_tol', None) != self.line_tol:
            nbins = max(1, int(np.ceil(np.pi * 2 * self.radius / (0.5 * self.line_tol))))
            order, self._bstart, self._rho, self._nrm, self._half = _line_index(self.t, self.v, nbins)
            npairs = len(self.t)
            self._order = order
            self._Xs = np.ascontiguousarray(self._X[order])
            self._gs = {k: np.ascontiguousarray(np.concatenate((x[order], x[order + npairs])))
                        for k, x in self.g.items()}
            self._index_tol = self.line_tol

    def _scores(self, scrambled, a, b, r1, r2, T, hkl_tol):
        ubis = self._orient(scrambled, a, b, r1, r2)
        out = np.zeros(len(a), np.int64)
        if len(a):
            self._index()
            _line_scores(ubis, np.ascontiguousarray(T, float), self._Xs, self._gs[scrambled], self._ok()[self._order],
                         2 if scrambled else 1, float(self.line_tol), float(hkl_tol),
                         self._bstart, self._rho, self._nrm, self._half, out)
        return out

    def run(self, r1, r2, hkl_tol):
        """ index on rings r1, r2; returns (minpks, trials, new grains) """
        if self.line_tol is None:
            self.measure_line_tol([(r1, r2)])
        a, b, d, sin, T = self._trials(r1, r2, False)
        an, bn, dn, sinn, Tn = self._trials(r1, r2, True)
        kn = np.arange(len(an))
        kn = self.rng.choice(kn, min(len(kn), len(a)), replace=False) if len(kn) else kn
        sn = self._scores(True, an[kn], bn[kn], r1, r2, Tn[kn], hkl_tol)
        minpks = int(sn.max()) if len(sn) else 0
        s = self._scores(False, a, b, r1, r2, T, hkl_tol)
        n0 = len(self.ubis)
        self._index()
        ok = self._ok()[self._order]
        m = np.zeros(len(self.t), np.uint8)
        ht2 = float(hkl_tol) ** 2
        for k in np.argsort(-s, kind='stable'):
            if s[k] <= minpks:
                break
            if self.claimed[self.pair[a[k]]] or self.claimed[self.pair[b[k]]]:
                continue
            ubi = self._orient(False, a[k:k + 1], b[k:k + 1], r1, r2)[0]
            m[:] = 0
            _line_hits(ubi, T[k].astype(float), self._Xs, self._gs[False], ok, 1, float(self.line_tol), ht2,
                       self._bstart, self._rho, self._nrm, self._half, m)
            q = np.flatnonzero(m)
            if len(q) > minpks:
                p = self._order[q]
                ok[q] = False
                self.claimed[p] = True
                self.pair_grain[p] = len(self.ubis)
                self.ubis.append(ubi)
                self.positions.append(T[k])
                self.npairs.append(len(p))
        return minpks, len(a), len(self.ubis) - n0


def _line_index(t, v, nbins):
    """ lines t + s v sorted for _line_hits: binned by the direction of their
    xy projection (nbins over 180 deg), sorted within a bin by their signed
    xy distance rho from the rotation axis; lines within 30 deg of vertical in
    a last bin. Returns order, bin starts, rho (sorted), bin centre normals,
    half bin width (radians). """
    vxy = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    phi = np.arctan2(v[:, 1], v[:, 0]) % np.pi
    rho = -np.sin(phi) * t[:, 0] + np.cos(phi) * t[:, 1]
    b = np.minimum((phi / np.pi * nbins).astype(int), nbins - 1)
    b[vxy < 0.5] = nbins
    order = np.lexsort((rho, b))
    bstart = np.searchsorted(b[order], np.arange(nbins + 2)).astype(np.int64)
    c = (np.arange(nbins) + 0.5) * np.pi / nbins
    nrm = np.ascontiguousarray(np.column_stack((-np.sin(c), np.cos(c))))
    return order, bstart, np.ascontiguousarray(rho[order]), nrm, 0.5 * np.pi / nbins


@numba.njit(cache=True)
def _hit(p, Tx, Ty, Tz, u, X, g, ok, rows, npairs, lt2, ht2):
    d0 = Tx - X[p, 0]
    d1 = Ty - X[p, 1]
    d2 = Tz - X[p, 2]
    s = d0 * X[p, 3] + d1 * X[p, 4] + d2 * X[p, 5]
    if d0 * d0 + d1 * d1 + d2 * d2 - s * s >= lt2 or not ok[p]:
        return False
    for r in range(rows):
        q = p + r * npairs
        gx, gy, gz = g[q, 0], g[q, 1], g[q, 2]
        h = u[0] * gx + u[1] * gy + u[2] * gz
        h -= np.rint(h)
        e2 = h * h
        if e2 >= ht2:
            continue
        h = u[3] * gx + u[4] * gy + u[5] * gz
        h -= np.rint(h)
        e2 += h * h
        if e2 >= ht2:
            continue
        h = u[6] * gx + u[7] * gy + u[8] * gz
        h -= np.rint(h)
        if e2 + h * h < ht2:
            return True
    return False


@numba.njit(cache=True)
def _line_hits(ubi, T, X, g, ok, rows, line_tol, ht2, bstart, rho, nrm, half, out):
    """ the pairs p (ok) whose line X[p] = (t0, v) passes within line_tol of
    T and whose g (rows p, p + npairs, ... up to rows) indexes with ubi (hkl
    error^2 < ht2): returns their count, and sets out[p] = 1 for them if out
    is not empty. X, g, ok, rho in the order of _line_index; only lines whose
    xy projection can pass within line_tol of T are tested. """
    u = ubi.ravel()
    Tx, Ty, Tz = T[0], T[1], T[2]
    npairs = X.shape[0]
    lt2 = line_tol * line_tol
    w = line_tol * (1 + 1e-9) + np.sqrt(Tx * Tx + Ty * Ty) * half
    mark = len(out) > 0
    nb = len(nrm)
    n = 0
    for k in range(nb + 1):
        lo, hi = bstart[k], bstart[k + 1]
        if k < nb:
            c = Tx * nrm[k, 0] + Ty * nrm[k, 1]
            r = rho[lo:hi]
            lo, hi = lo + np.searchsorted(r, c - w), lo + np.searchsorted(r, c + w)
        for p in range(lo, hi):
            if _hit(p, Tx, Ty, Tz, u, X, g, ok, rows, npairs, lt2, ht2):
                if mark:
                    out[p] = 1
                n += 1
    return n


@numba.njit(parallel=True, cache=True)
def _line_scores(ubis, T, X, g, ok, rows, line_tol, hkl_tol, bstart, rho, nrm, half, out):
    """ out[i]: the count of _line_hits for trial i """
    ht2 = hkl_tol * hkl_tol
    none = np.zeros(0, np.uint8)
    for i in numba.prange(len(ubis)):
        out[i] = _line_hits(np.ascontiguousarray(ubis[i]), T[i], X, g, ok, rows, line_tol, ht2, bstart, rho, nrm, half, none)


# ---------------------------------------------------------------- positions

def fit_position(x1, o1, x2, o2, pars, niter=8, h=100.0):
    """
    Translation t minimising sum |g_s1(t) + g_s2(t)|^2 over the pairs of one
    grain (Gauss-Newton); pairs further than 3 robust widths (1.4826 MAD of
    |g_s1 + g_s2|) are dropped as the fit proceeds.
    Returns t, rms |g_s1 + g_s2| of the pairs kept, number kept.
    """
    t = np.zeros(3)
    keep = np.ones(len(x1), bool)
    for _ in range(niter):
        r = _pair_residual(x1[keep], o1[keep], x2[keep], o2[keep], t, pars)
        J = np.empty((keep.sum(), 3, 3))
        for k in range(3):
            dt = np.zeros(3)
            dt[k] = h
            J[:, :, k] = (_pair_residual(x1[keep], o1[keep], x2[keep], o2[keep], t + dt, pars) -
                          _pair_residual(x1[keep], o1[keep], x2[keep], o2[keep], t - dt, pars)) / (2 * h)
        t = t - np.linalg.lstsq(J.reshape(-1, 3), r.reshape(-1), rcond=None)[0]
        res = np.sqrt((_pair_residual(x1, o1, x2, o2, t, pars) ** 2).sum(axis=1))
        new = res < 3 * 1.4826 * np.median(res[keep])
        if new.sum() < 3:
            break
        keep = new
    return t, np.sqrt(np.mean(res[keep] ** 2)), int(keep.sum())


def fit_positions(cf, i1, i2, pair_grain, ngrains):
    """ fit_position for each grain from the pairs labelled with it
    (pair_grain[k] = grain of pair k, -1 for none). Grains with fewer than 3
    pairs get no fit (nused 0). Returns (ngrains, 3) translations, rms
    |g_s1 + g_s2| and pairs used per grain. """
    pars = cf.parameters.parameters
    xyz = lab_xyz(cf)
    t = np.zeros((ngrains, 3))
    rms = np.full(ngrains, np.nan)
    nused = np.zeros(ngrains, int)
    for g in range(ngrains):
        k = np.flatnonzero(pair_grain == g)
        if len(k) >= 3:
            t[g], rms[g], nused[g] = fit_position(xyz[i1[k]], cf.omega[i1[k]], xyz[i2[k]], cf.omega[i2[k]], pars)
    return t, rms, nused
