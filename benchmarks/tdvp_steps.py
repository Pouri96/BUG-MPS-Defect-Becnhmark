"""One- and two-site TDVP sweeps, shared by the figure scripts.

Both are a single left-to-right sweep over the chain, so that they are directly
comparable with the single left-to-right sweep of BUG-MPS
(bug_mps_checks.bug_mps_step).  Local equations are solved exactly with a dense
matrix exponential, as everywhere else in these scripts.

Both sweeps enter assuming the orthogonality centre sits at site 0 and every other
tensor is right-isometric, but a left-to-right sweep leaves the centre at the last
site.  Each step therefore ends with right_canonicalise, which restores that input
gauge so the steps compose.  It is a gauge change only: the state, and hence the
one-step error, is unchanged.  (bug_mps_step needs no such fix -- its backward SVD
sweep already returns the centre to site 0.)

  tdvp_step   one-site TDVP, the projector-splitting integrator of Lubich and
              Oseledets: forward K-step at each site, backward C-step on the
              bond matrix in between.  The rank never changes.

  tdvp2_step  two-site TDVP: forward evolution of the two-site tensor at each
              bond, split by an SVD, then a backward one-site step on the new
              centre.  With no truncation the bond grows to at most
              min(r_{i-1} d_i, d_{i+1} r_{i+1}), i.e. to at most 2r for qubits,
              the same budget a BUG-MPS step has.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import qr, svd
from scipy.linalg import expm

from bug_mps_checks import (
    ten_R, mat_R, iota_matrix, iota2_matrix, left_blocks, right_blocks,
    right_canonicalise, expm_apply,
)


def iota_bond_matrix(Lam_new, P_old):
    """Embed a 0-site bond matrix R (rl x rr) into H:
    vec(R) -> kron(Lam_new, P_old^T) vec(R). Same construction as iota_matrix
    but with no physical site in between."""
    return np.kron(Lam_new, P_old.T)


def tdvp_step(mps, F, dt, backward=True):
    """One-site TDVP: K-step forward by dt, thin QR at the SAME rank (no
    enlarge), then (if backward) a C-step evolving the bond matrix by -dt
    before folding it into the next site. backward=False is a negative
    control: a forward-only sweep is not a consistent splitting of the
    tangent-space projection and gives a nonsense, rank-independent floor."""
    N = len(mps)
    cur = [A.copy() for A in mps]
    for k in range(N - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        Meff = Ik.conj().T @ F @ Ik
        XK = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)
        Q, R = qr(XK, mode="reduced")          # same rank rr, no enlarge
        cur[k] = ten_R(Q, rl, d)                # install new site tensor first

        if backward:
            Lam_new = left_blocks(cur, k + 1)   # sites 0..k, through the new Q
            P_old = right_blocks(mps, k)        # sites k+1..N-1, old/untouched
            Ib = iota_bond_matrix(Lam_new, P_old)
            Meff_bond = Ib.conj().T @ F @ Ib
            R = expm_apply(-dt * Meff_bond, R.reshape(-1)).reshape(rr, rr)

        cur[k + 1] = np.einsum("ab,bcp->acp", R, cur[k + 1])

    k = N - 1                                   # final site: forward K-step only
    rl, rr, d = cur[k].shape
    Ik = iota_matrix(cur, k)
    XC = mat_R(cur[k])
    Meff = Ik.conj().T @ F @ Ik
    XC1 = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)
    cur[k] = ten_R(XC1, rl, d)
    return right_canonicalise(cur)


def two_site_tensor(A, B):
    """Contract neighbouring tensors into Mat_k(Theta), rows (alpha_{k-1}, p_k),
    columns (p_{k+1}, alpha_{k+1}); A : (rl, rm, dk), B : (rm, rr, dk1)."""
    rl, rm, dk = A.shape
    _, rr, dk1 = B.shape
    T = np.einsum("agp,gbq->apqb", A, B)        # (rl, dk, dk1, rr)
    return T.reshape(rl * dk, dk1 * rr)


def tdvp2_step(mps, F, dt, theta=0.0, rmax=None, backward=True, tol=1e-12):
    """Two-site TDVP, one left-to-right sweep. theta is a relative singular
    value tolerance for the split (theta = 0 keeps every nonzero value, so the
    bond grows as far as the two-site tensor allows); rmax caps the bond."""
    N = len(mps)
    cur = [A.copy() for A in mps]
    for k in range(N - 1):
        rl, _, dk = cur[k].shape
        _, rr, dk1 = cur[k + 1].shape
        I2 = iota2_matrix(cur, k)
        Theta = two_site_tensor(cur[k], cur[k + 1])
        Meff2 = I2.conj().T @ F @ I2
        Theta = expm_apply(dt * Meff2, Theta.reshape(-1)).reshape(rl * dk, dk1 * rr)

        U, s, Vh = svd(Theta, full_matrices=False)
        keep = int(np.sum(s > max(theta, tol) * s[0])) if s[0] > 0 else 1
        keep = max(keep, 1)
        if rmax is not None:
            keep = min(keep, rmax)
        U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
        cur[k] = ten_R(U, rl, dk)                               # left-isometric
        C = (s[:, None] * Vh).reshape(keep, dk1, rr).transpose(0, 2, 1)
        cur[k + 1] = C

        if backward and k < N - 2:
            rl1, rr1, d1 = cur[k + 1].shape
            Ik = iota_matrix(cur, k + 1)
            Meff = Ik.conj().T @ F @ Ik
            XC = mat_R(cur[k + 1])
            XC = expm_apply(-dt * Meff, XC.reshape(-1)).reshape(rl1 * d1, rr1)
            cur[k + 1] = ten_R(XC, rl1, d1)
    return right_canonicalise(cur)
