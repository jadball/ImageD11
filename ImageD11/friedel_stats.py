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
- pair_peaks : mutual nearest partners in (omega, eta, 2theta, log I), scales
  from the pairs, cut where pairs stop outnumbering chance (partners searched
  at a wrong omega)
- pair_colfile : the paired peaks with the pair g-vectors
- ring_windows : ring centre, width and d* window from the paired peaks
- index_pairs : the ImageD11 indexer, trial orientations strongest first,
  minpks from the same search on scrambled peaks (each g turned about the
  rotation axis by a random angle)
- fit_positions : grain positions from their pairs
"""

from __future__ import print_function, division

import numpy as np
import scipy.spatial
from ImageD11 import cImageD11, transform, indexing

# (omega, eta, tth, log10 I); omega and eta are periodic
_PERIODIC = np.array([True, True, False, False])
NAMES = ('omega', 'eta', 'tth', 'logI')


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


def pair_peaks(cf, null_offsets=(7.1, -7.1, 13.3, -13.3), niter=4):
    """
    Friedel pairs in a columnfile (omega, eta, tth, sum_intensity, sc, fc).

    Each peak's partner is the nearest peak to where its -g diffracts
    (predict_partners) with the same log I. Distances are in units of the
    spread of each coordinate over the pairs (1.4826 MAD, centred on the
    median offset), re-measured on the pairs inside the cut until it settles:
    no weights to set.

    Cut: partners are also searched null_offsets degrees away in omega (far
    beyond the omega spread of real pairs), where only chance finds a peak.
    The cut r maximises N(d < r) - 2 N_null(d < r): beyond it a wider window
    adds more chance partners than real ones.

    Returns a dict: i1, i2 (pairs: mutual nearest, within the cut), cut, scale,
    centre, and r, n_obs, n_null (the curves), d (all nearest distances).
    """
    X = _space(cf)
    n = len(X)
    pred = predict_partners(cf)
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
        dn = np.concatenate([_nearest(X, _space(cf, pred, o) + centre, scale)[0][valid]
                             for o in null_offsets])
        r = np.linspace(0, np.percentile(d[valid], 99), 500)
        n_obs = np.searchsorted(np.sort(d), r)
        n_null = np.searchsorted(np.sort(dn), r) / len(null_offsets)
        cut = r[np.argmax(n_obs - 2 * n_null)]
    i = np.arange(n)
    keep = (j[j] == i) & (d < cut) & (i < j)
    return {'i1': i[keep], 'i2': j[keep], 'cut': cut, 'scale': scale,
            'centre': centre, 'r': r, 'n_obs': n_obs, 'n_null': n_null, 'd': d}


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
    x1, x2, o1, o2 = xyz[i1], xyz[i2], cf.omega[i1], cf.omega[i2]
    t = np.zeros((len(i1), 3))
    for _ in range(niter):
        r = _pair_residual(x1, o1, x2, o2, t, pars)
        J = np.empty((len(i1), 3, 3))
        for k in range(3):
            dt = np.zeros(3)
            dt[k] = h
            J[:, :, k] = (_pair_residual(x1, o1, x2, o2, t + dt, pars) -
                          _pair_residual(x1, o1, x2, o2, t - dt, pars)) / (2 * h)
        U, S, Vt = np.linalg.svd(J)
        c = np.einsum('nji,nj->ni', U[:, :, :2], r) / S[:, :2]
        t -= np.einsum('nij,ni->nj', Vt[:, :2, :], c)
    g1 = gv_rows(x1, o1, t, pars)
    g2 = gv_rows(x2, o2, t, pars)
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

def _orient_scores(ind, hits, tol):
    """ npks of the orientation made by each hit (i, j) on rings 1, 2 """
    gv = ind.gvflat
    out = np.zeros(len(hits), int)
    for k, (_, i, j) in enumerate(hits):
        ind.unitcell.orient(ind.ring_1, ind.gv[i], ind.ring_2, ind.gv[j], verbose=0)
        out[k] = cImageD11.score(ind.unitcell.UBI, gv, tol)
    return out


def _scrambled(ind, rng):
    """ a copy of the indexer with each g-vector turned about the rotation
    axis by a random angle (same |g| and g_z, so the same rings and detector
    coverage): no lattice left """
    a = rng.uniform(0, 2 * np.pi, len(ind.gv))
    c, s = np.cos(a), np.sin(a)
    g = ind.gv
    v = np.column_stack([c * g[:, 0] - s * g[:, 1], s * g[:, 0] + c * g[:, 1], g[:, 2]])
    s = indexing.indexer(unitcell=ind.unitcell, gv=v, wavelength=ind.wavelength)
    s.ds_tol = ind.ds_tol
    s.cosine_tol = ind.cosine_tol
    s.assigntorings()
    return s


def _ring_numbers(ringds, dstars):
    r = np.asarray(ringds)
    return [int(np.argmin(np.abs(r - d))) for d in dstars]


def index_pairs(cfp, ucell, gen_ds, score_ds, dstol, hkl_tols, cosine_tol,
                max_grains=1000, seed=0):
    """
    Index the pair g-vectors with the ImageD11 indexer.

    gen_ds, score_ds : d* of the rings to make orientations from and to score
    on (the indexer numbers its rings with tolerance dstol, so they are given
    by d*, not by number).
    For each hkl_tol and pair of rings in gen_ds, the trial orientations
    (indexer.find) are scored and taken strongest first (indexer.scorethem,
    which also rejects an orientation whose peaks are mostly taken).
    minpks: the highest score of the same search on scrambled g-vectors
    (each turned about the rotation axis by a random angle), so chance alone
    is not expected to pass.

    Returns the indexer (ubis, ga, colfile) and a list of (hkl_tol, ring_1,
    ring_2, minpks, trials, new grains), rings numbered as in the indexer.
    """
    ind = indexing.indexer_from_colfile_and_ucell(cfp, ucell=ucell)
    ind.ds_tol = dstol
    ind.assigntorings()
    keep = np.isin(ind.ra, _ring_numbers(ind.unitcell.ringds, score_ds))
    ind = indexing.indexer_from_colfile_and_ucell(cfp.copyrows(keep), ucell=ucell)
    ind.ds_tol = dstol
    ind.cosine_tol = cosine_tol
    ind.max_grains = max_grains
    ind.assigntorings()
    rings_gen = _ring_numbers(ind.unitcell.ringds, gen_ds)
    null = _scrambled(ind, np.random.RandomState(seed))
    log = []
    threads = cImageD11.cimaged11_omp_get_max_threads()
    cImageD11.cimaged11_omp_set_num_threads(1)
    try:
        for tol in hkl_tols:
            ind.hkl_tol = null.hkl_tol = tol
            for a in range(len(rings_gen)):
                for b in range(a, len(rings_gen)):
                    for x in (ind, null):
                        x.ring_1, x.ring_2 = rings_gen[a], rings_gen[b]
                        x.find()
                    nscore = _orient_scores(null, null.hits, tol)
                    ind.minpks = int(nscore.max()) if len(nscore) else 0
                    s = _orient_scores(ind, ind.hits, tol)
                    order = np.argsort(s, kind='stable')
                    ntrial = len(ind.hits)
                    # scorethem pops from the end: strongest first
                    ind.hits = [ind.hits[k] for k in order if s[k] > ind.minpks]
                    nb4 = len(ind.ubis)
                    if ind.hits:
                        ind.scorethem()
                    log.append((tol, rings_gen[a], rings_gen[b], ind.minpks, ntrial,
                                len(ind.ubis) - nb4))
    finally:
        cImageD11.cimaged11_omp_set_num_threads(threads)
    return ind, log


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
