"""
Peak-count cuts and grain acceptance for grid indexing, measured from the data.

- grid_null_scores : what random orientations score in the grid indexer
- match_grains, null_matches, accept_grains : keep the grains that explain
  more unclaimed peaks than chance would
- refine_grains : makemap.py in a function, dropping grains whose fit fails
"""

from __future__ import print_function, division

import os
import sys
import multiprocessing
import numpy as np
from ImageD11 import cImageD11, transform, transformer, unitcell, grain, simplex
from ImageD11.refinegrains import refinegrains
from ImageD11.grid_index_parallel import gvectors_at, domap_robust


def random_rotations(n, rng):
    """ n random rotation matrices, uniform over SO(3) (unit quaternions) """
    q = rng.normal(size=(n, 4))
    q /= np.sqrt((q * q).sum(axis=1))[:, None]
    w, x, y, z = q.T
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
                    ).transpose(2, 0, 1)


class PeakGeometry(object):
    """ Lab positions and omega of the peaks in a columnfile, to compute
    g-vectors for a grain at any translation """

    def __init__(self, cf):
        pars = cf.parameters.parameters
        self.xyz = np.ascontiguousarray(
            transform.compute_xyz_lab([cf.sc, cf.fc], **pars).T)
        self.omega = np.ascontiguousarray(cf.omega, dtype=float)
        self.args = (float(pars['omegasign']), float(pars['wavelength']),
                     float(pars['wedge']), float(pars['chi']))
        self.n = len(self.omega)

    def gv(self, t, out=None):
        """ g-vectors (n, 3) for a grain at translation t """
        if out is None:
            out = np.empty((self.n, 3), float)
        omegasign, wvln, wedge, chi = self.args
        cImageD11.compute_gv(self.xyz, self.omega, omegasign, wvln, wedge, chi,
                             np.asarray(t, float), out)
        return out


# ---------------------------------------------------------------- grid null

def _grid_null_job(args):
    fltfile, parfile, gridpars, t, n, seed = args
    cImageD11.cimaged11_omp_set_num_threads(1)
    tr = transformer.transformer()
    tr.loadfiltered(fltfile)
    tr.loadfileparameters(parfile)
    col = tr.colfile
    for name, val in (("drlv2", 1.0), ("labels", -1.0)):
        if name not in col.titles:
            col.addcolumn(np.full(col.nrows, val), name)
    gp = dict(gridpars)
    gp['UC'] = unitcell.unitcell_from_parameters(tr.parameterobj)
    gp['NPKS'] = -1
    col, gve = gvectors_at(tr, t, gp)
    gv = np.ascontiguousarray(gve.T)
    B = gp['UC'].B
    ubis = [np.linalg.inv(U.dot(B)) for U in random_rotations(n, np.random.RandomState(seed))]
    s_index = [cImageD11.score(u, gv, gp['TOLSEQ'][0]) for u in ubis]
    grs = domap_robust(tr.parameterobj, col, [grain.grain(u, t) for u in ubis], gp)
    return s_index, [int(g.npks) for g in grs]


def grid_null_scores(fltfile, parfile, gridpars, points, n_per_point, nproc, seed=0):
    """
    Random orientations at a few grid points, scored as the grid indexer
    scores its candidates:
      index : cImageD11.score at TOLSEQ[0] (the doindex test, NPKS_INDEX)
      domap : npks after domap refinement (the domap test, NPKS)
    Returns two arrays: index scores, domap scores.
    """
    chunk = 50
    jobs = []
    for i, t in enumerate(points):
        for j in range(0, n_per_point, chunk):
            jobs.append((fltfile, parfile, gridpars, tuple(t),
                         min(chunk, n_per_point - j), seed + 1000 * i + j))
    p = multiprocessing.Pool(nproc)
    try:
        res = p.map(_grid_null_job, jobs, chunksize=1)
    finally:
        p.close()
        p.join()
    return (np.concatenate([r[0] for r in res]),
            np.concatenate([r[1] for r in res]))


# ------------------------------------------ scoring against all peaks (workers)

_G = {}


def _init_worker(geom):
    cImageD11.cimaged11_omp_set_num_threads(1)
    _G['geom'] = geom
    _G['gv'] = np.empty((geom.n, 3), float)
    _G['d2'] = np.empty(geom.n, float)
    _G['lab'] = np.empty(geom.n, np.int32)


def _assign(ubi, t, tol):
    """ indices and drlv2 of peaks within hkl_tol of an integer hkl """
    gv = _G['geom'].gv(t, out=_G['gv'])
    d2, lab = _G['d2'], _G['lab']
    d2[:] = 1
    lab[:] = -1
    cImageD11.score_and_assign(np.ascontiguousarray(ubi, float), gv, tol, d2, lab, 0)
    idx = np.flatnonzero(lab == 0)
    return idx, d2[idx].copy()


def _match_job(args):
    ubi, t, tol = args
    return _assign(ubi, t, tol)[0]


def _null_job(args):
    t, B, n, tolseq, tol, seed = args
    gv = _G['geom'].gv(t, out=_G['gv'])
    out = []
    for U in random_rotations(n, np.random.RandomState(seed)):
        u = np.linalg.inv(U.dot(B))
        for tl in tolseq:
            for _ in range(3):
                cImageD11.score_and_refine(u, gv, tl)
        if np.linalg.det(u) > 0:
            out.append(_assign(u, t, tol)[0])
        else:
            out.append(np.zeros(0, int))
    return out


def _pool_map(geom, func, jobs, nproc):
    p = multiprocessing.Pool(nproc, initializer=_init_worker, initargs=(geom,))
    try:
        return p.map(func, jobs, chunksize=max(1, len(jobs) // (4 * nproc)))
    finally:
        p.close()
        p.join()


def match_grains(geom, grains, tol, nproc):
    """ For each grain: indices of the peaks it indexes at hkl_tol tol,
    at its own position """
    jobs = [(g.ubi, np.asarray(g.translation, float), tol) for g in grains]
    return _pool_map(geom, _match_job, jobs, nproc)


def null_matches(geom, translations, B, tolseq, tol, n_per_position, nproc, seed=0):
    """
    Random orientations placed at the given translations (grain positions),
    refined against all peaks through tolseq like a candidate, then matched
    at hkl_tol tol. Returns a list of peak index arrays.
    """
    jobs = [(np.asarray(t, float), B, n_per_position, list(tolseq), tol, seed + i)
            for i, t in enumerate(translations)]
    res = _pool_map(geom, _null_job, jobs, nproc)
    return [m for r in res for m in r]


def greedy_accept(matched, cut, claimed):
    """ Strongest first; a grain is accepted if more than cut of its peaks are
    not claimed yet, then it claims them. claimed is updated in place.
    Returns accepted indices and the unclaimed count of every grain when its
    turn came. """
    order = np.argsort([-len(m) for m in matched], kind='stable')
    accepted = []
    unclaimed = np.zeros(len(matched), int)
    for j in order:
        unclaimed[j] = (~claimed[matched[j]]).sum()
        if unclaimed[j] > cut:
            claimed[matched[j]] = True
            accepted.append(j)
    return accepted, unclaimed


def accept_grains(matched, nulls, claimed, max_false=1.0):
    """
    Choose the unclaimed-peak cut from the null and accept grains.

    matched : peak indices per candidate; nulls : peak indices per random lattice
    claimed : bool per peak, claimed by grains accepted earlier (not modified)
    The null lattices are counted against the peaks claimed by a provisional
    accepted set (cut at the 99.9th percentile of their raw counts). With N
    candidates, the cut c is the smallest with N * P(null unclaimed > c) <= max_false.

    Returns accepted indices, claimed (new array), cut, expected false grains,
    unclaimed counts of the candidates, unclaimed counts of the null.
    """
    nraw = np.array([len(m) for m in nulls])
    cl = claimed.copy()
    greedy_accept(matched, np.percentile(nraw, 99.9), cl)
    null_un = np.array([(~cl[m]).sum() for m in nulls])
    N = len(matched)
    cut = 0
    while N * (null_un > cut).mean() > max_false:
        cut += 1
    cl = claimed.copy()
    accepted, unclaimed = greedy_accept(matched, cut, cl)
    return accepted, cl, cut, N * (null_un > cut).mean(), unclaimed, null_un


class _RefineDropFailed(refinegrains):
    """ refinegrains.refinepositions, but a grain whose fit fails (a left
    handed UBI from a fit to too few peaks) goes to self.failed instead of
    stopping the run """

    def refinepositions(self, quiet=True, maxiters=100):
        self.assignlabels()
        self.failed = []
        tolcache = self.tolerance
        self.tolerance = 1.0
        for key in sorted(self.grains.keys()):
            self.grains_to_refine = [key]
            self.parameterobj.varylist = ['t_x', 't_y', 't_z']
            self.set_translation(key[0], key[1])
            guess = self.parameterobj.get_variable_values()
            inc = self.parameterobj.get_variable_stepsizes()
            try:
                simplex.Simplex(self.gof, guess, inc).minimize(maxiters=maxiters, monitor=1)
                for i, name in enumerate(['t_x', 't_y', 't_z']):
                    self.grains[key].translation[i] = self.parameterobj.parameters[name]
                self.refine(self.grains[key].ubi, quiet=False)
            except AssertionError:
                self.failed.append(key[0])
        self.tolerance = tolcache


def refine_grains(parfile, colfile, grains, tol, omega_slop, symmetry):
    """
    makemap.py in a function: refine the positions and UBIs of grains against
    the peaks in colfile at hkl tolerance tol. A grain whose fit fails is
    dropped and the rest are refined again (peak assignment depends on all
    grains). colfile gets the makemap columns (labels, h, k, l, drlv2, ...).
    Returns (refined grains, indices into grains of those kept).
    """
    # columns refinegrains.loadfiltered adds
    if 'drlv2' not in colfile.titles:
        colfile.addcolumn(np.ones(colfile.nrows), 'drlv2')
    if 'labels' not in colfile.titles:
        colfile.addcolumn(np.full(colfile.nrows, -1.), 'labels')
    if 'sc' not in colfile.titles:
        colfile.addcolumn(colfile.xc.copy(), 'sc')
    if 'fc' not in colfile.titles:
        colfile.addcolumn(colfile.yc.copy(), 'fc')
    kept = list(range(len(grains)))
    ss = sys.stdout
    while True:
        sys.stdout = open(os.devnull, 'w')
        try:
            o = _RefineDropFailed(tolerance=tol, intensity_tth_range=(0, 180),
                                  OmFloat=omega_slop > 0, OmSlop=omega_slop)
            o.loadparameters(parfile)
            o.scannames = ['internal']
            o.scantitles = colfile.titles
            o.scandata['internal'] = colfile
            for i in kept:
                o.grainnames.append(i)
                o.ubisread[i] = grains[i].ubi
                o.translationsread[i] = grains[i].translation
            if symmetry != 'triclinic':
                o.makeuniq(symmetry)
            o.generate_grains()
            o.refinepositions()
            if not o.failed:
                o.savegrains(os.devnull, sort_npks=False)
        finally:
            sys.stdout.close()
            sys.stdout = ss
        if not o.failed:
            return [o.grains[(i, 'internal')] for i in kept], kept
        print('%d grains dropped: refinement failed (left handed UBI)' % len(o.failed))
        failed = set(o.failed)
        kept = [i for i in kept if i not in failed]


def _refine_init(args):
    cImageD11.cimaged11_omp_set_num_threads(1)
    _G['refine'] = args


def _refine_job(ubis_translations):
    parfile, colfile, tol, omega_slop, symmetry = _G['refine']
    grains = [grain.grain(u, t) for u, t in ubis_translations]
    out, kept = refine_grains(parfile, colfile, grains, tol, omega_slop, symmetry)
    return [(g.ubi, np.array(g.translation), g.npks, g.intensity_info) for g in out], kept


def refine_grains_parallel(parfile, colfile, grains, tol, omega_slop, symmetry, nproc):
    """
    refine_grains on chunks of grains in parallel. Peaks are assigned among
    the grains of one chunk only, so two grains in different chunks can share
    peaks: for candidates before an acceptance step that settles shared peaks.
    colfile is not changed. Returns (refined grains, indices into grains kept).
    """
    chunk = max(1, len(grains) // (4 * nproc))
    starts = list(range(0, len(grains), chunk))
    jobs = [[(g.ubi, np.asarray(g.translation, float)) for g in grains[s:s + chunk]] for s in starts]
    p = multiprocessing.Pool(nproc, initializer=_refine_init,
                             initargs=((parfile, colfile, tol, omega_slop, symmetry),))
    try:
        res = p.map(_refine_job, jobs, chunksize=1)
    finally:
        p.close()
        p.join()
    out, kept = [], []
    for s, (gl, k) in zip(starts, res):
        for (u, t, npks, info), i in zip(gl, k):
            g = grain.grain(u, t)
            g.npks, g.intensity_info = npks, info
            out.append(g)
            kept.append(s + i)
    return out, kept
