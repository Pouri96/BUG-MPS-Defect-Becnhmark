"""
Numerical checks for the robustness argument of BUG-MPS (replaces conj:eta_eps).

Reuses the dense machinery of bug_mps_checks.py; run from the repository root:

    python benchmarks/robustness_checks.py

Notation (as in the note).  During the sweep, at bond k:
  Lambda_k   augmented left block (Lhat^[0] ... Lhat^[k-1]), an isometry;
  P_k        frozen right block of Psi^0 (R^[k+1] ... R^[N-1]), P_k P_k^* = I;
  Pihat_k    orthogonal projector onto the augmented bond space Vhat_k;
  E_k        Z^<k>  ->  (Lambda_k (x) I) (I - Pihat_k) (Lambda_k (x) I)^* Z^<k>,
  Q_k        Z^<k>  ->  Z^<k> (I - P_k^* P_k).
E_k acts on sites 0..k and Q_k on sites k+1..N-1, so they commute, and
  I - P_Shat = sum_k E_k            (telescoping, Lambda_{k+1} = (Lambda_k (x) I) Q_r),
  D := sum_k E_k Q_k                is an orthogonal projector.

Checks:
  T12 bond-splitting lemma  E_k Q_k X = 0 for every tangent vector X at Psi^0, every
                            k and every dt; with two negative controls showing
                            that each of its two factors is needed.
  T13 dt -> 0 structure     (I - P_Shat) F Psi^0 = D R_0, R_0 the normal component,
                            hence eta_0 <= eps; D is an orthogonal projector that
                            annihilates the tangent space; for a nearest-neighbour
                            generator the bond-k part of the defect is exactly the
                            two-site residual at bond k.
  T14 finite dt             eta^2 = ||D R_0||^2 + sum_k ||E_k P_k^* P_k G||^2 exactly,
                            and eta <= eps + sqrt(N-1) (dt/2) ||F||^2 ||Psi^0|| e^{dt||F||}.
  T15 two-site augmentation adding the two-site residual at every bond removes the
                            first-order defect for a nearest-neighbour generator (and
                            in general not for a dense one).
  T16 loss below corner     at every bond, the part of the field lost there is at most
                            the corner (normal) block of the field at that cut.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.linalg import qr, norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bug_mps_checks import (  # noqa: E402
    mat_R, ten_R, random_tensor, random_mps, mps_to_vector, left_blocks,
    right_blocks, iota_matrix, orth_basis, proj, bug_mps_step, tangent_basis,
    two_local_generator,
)

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# sweeps that expose Lambda_k and Pihat_k
# --------------------------------------------------------------------------- #


def limit_sweep(mps, F, tol=1e-10, retain=True):
    """K-sweep in the limit dt -> 0 (as eta_limit in bug_mps_checks.py): bond k is
    augmented with Mat_R(iota_k^* F Psi^0).  retain=False drops V_k^0 from the
    augmentation (negative control only; the result is then not a gauge of Psi^0)."""
    N = len(mps)
    cur = [A.copy() for A in mps]
    Qrs = []
    for k in range(N - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        XK = ((Ik.conj().T @ F @ Ik) @ XC.reshape(-1)).reshape(rl * d, rr)
        Q, U = qr(XC, mode="reduced")
        Qr = orth_basis(np.hstack([Q, XK]) if retain else XK, tol=tol)
        Qrs.append(Qr)
        T = Qr.conj().T @ Q
        cur[k] = ten_R(Qr, rl, d)
        Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
        cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
    return cur, Qrs


def finite_sweep(mps, F, dt):
    _, rec = bug_mps_step(mps, F, dt, theta=0.0)
    N = len(mps)
    return rec["pre_trunc"], [rec["Qr"][k] for k in range(N - 1)]


class Bonds:
    """The operators E_k, Q_k, P_k^* P_k of one sweep, applied to vectors on H."""

    def __init__(self, mps0, cur, Qrs):
        self.ds = [A.shape[2] for A in mps0]
        self.N = len(mps0)
        self.L, self.Pi, self.P = [], [], []
        for k in range(self.N - 1):
            Lam = left_blocks(cur, k)
            self.L.append(np.kron(Lam, np.eye(self.ds[k])))
            self.Pi.append(proj(Qrs[k]))
            self.P.append(right_blocks(mps0, k))

    def _mat(self, z, k):
        return z.reshape(int(np.prod(self.ds[: k + 1])), -1)

    def E(self, z, k):
        Z = self._mat(z, k)
        A = self.L[k]
        Y = A.conj().T @ Z
        return (A @ (Y - self.Pi[k] @ Y)).reshape(-1)

    def Q(self, z, k):
        Z = self._mat(z, k)
        P = self.P[k]
        return (Z - (Z @ P.conj().T) @ P).reshape(-1)

    def PP(self, z, k):
        Z = self._mat(z, k)
        P = self.P[k]
        return ((Z @ P.conj().T) @ P).reshape(-1)

    def D(self, z):
        return sum(self.E(self.Q(z, k), k) for k in range(self.N - 1))

    def bond_defect(self, z, k):
        return norm(self.E(self.Q(z, k), k))

    def two_site_residual(self, g, k):
        """|| (I - Pihat_k) Mat_k(iota_{k,k+1}^* g) ||, P_{k+1} frozen."""
        Z = self._mat(g, k)                                   # rows p_0..p_k
        dk1 = self.ds[k + 1]
        P1 = self.P[k + 1] if k + 1 < self.N - 1 else np.ones((1, 1))
        Z3 = Z.reshape(Z.shape[0], dk1, -1) @ P1.conj().T     # (rows, d_{k+1}, r_{k+1})
        Y = self.L[k].conj().T @ Z3.reshape(Z.shape[0], -1)
        return norm(Y - self.Pi[k] @ Y)


def random_instance(rng, Nmin=2, Nmax=8, dmax=3, rmax=3, Dmax=256, local=None):
    while True:
        N = int(rng.integers(Nmin, Nmax + 1))
        ds = [int(x) for x in rng.integers(2, dmax + 1, size=N)]
        if int(np.prod(ds)) <= Dmax:
            break
    rs = [int(x) for x in rng.integers(1, rmax + 1, size=N - 1)]   # non-uniform
    mps = random_mps(rng, ds, rs)
    D = int(np.prod(ds))
    if local is None:
        local = rng.random() < 0.5
    if local:
        F = two_local_generator(rng, ds)
    else:
        F = random_tensor(rng, (D, D))
    if rng.random() < 0.3:                                # anti-Hermitian, -iH
        F = 0.5 * (F - F.conj().T)
    return mps, F, ds, local


def eps_and_normal(mps, F):
    psi0 = mps_to_vector(mps)
    Tb = tangent_basis(mps)
    g = F @ psi0
    R0 = g - Tb @ (Tb.conj().T @ g)
    return psi0, Tb, g, R0, norm(R0)


# --------------------------------------------------------------------------- #
# T12  bond-splitting lemma
# --------------------------------------------------------------------------- #


def t12_bond_splitting(rng):
    worst, worst_noQ, worst_noret, ninst, nvec = 0.0, 0.0, 0.0, 0, 0
    for it in range(120):
        mps, F, ds, _ = random_instance(rng)
        _, Tb, _, _, _ = eps_and_normal(mps, F)
        dt = [None, 1.0, 1e-1, 1e-3][it % 4]
        cur, Qrs = limit_sweep(mps, F) if dt is None else finite_sweep(mps, F, dt)
        B = Bonds(mps, cur, Qrs)
        X = np.hstack([Tb, Tb @ random_tensor(rng, (Tb.shape[1], 4))])
        for j in range(X.shape[1]):
            x = X[:, j] / norm(X[:, j])
            for k in range(B.N - 1):
                worst = max(worst, norm(B.E(B.Q(x, k), k)))
                worst_noQ = max(worst_noQ, norm(B.E(x, k)))
            nvec += 1
        ninst += 1
        # negative control: augment with the predictor only, without V_k^0
        if B.N >= 3 and dt is None:
            curn, Qrn = limit_sweep(mps, F, retain=False)
            Bn = Bonds(mps, curn, Qrn)
            for j in range(Tb.shape[1]):
                x = Tb[:, j]
                for k in range(Bn.N - 1):
                    worst_noret = max(worst_noret, norm(Bn.E(Bn.Q(x, k), k)))
    report("T12a bond-splitting lemma  E_k Q_k X = 0 for all X in T_Psi M_r, all k, dt",
           worst < 1e-10,
           f"{ninst} instances (N=2..8, d in {{2,3}}, non-uniform r <= 3, dt in "
           f"{{0, 1, 0.1, 1e-3}}), {nvec} unit tangent vectors, max |E_k Q_k X| = "
           f"{worst:.1e}")
    report("T12b control: without the right factor Q_k the identity fails",
           worst_noQ > 1e-2, f"max |E_k X| = {worst_noQ:.2f}")
    report("T12c control: without retaining V_k^0 in Vhat_k the identity fails",
           worst_noret > 1e-2, f"max |E_k Q_k X| = {worst_noret:.2f}")


# --------------------------------------------------------------------------- #
# T13  the dt -> 0 structure of the defect
# --------------------------------------------------------------------------- #


def t13_limit_structure(rng):
    w_id, w_DR, w_proj, w_DT, w_ts = 0.0, 0.0, 0.0, 0.0, 0.0
    ratio, nviol, n, n_local = 0.0, 0, 0, 0
    for it in range(200):
        mps, F, ds, local = random_instance(rng)
        psi0, Tb, g, R0, eps = eps_and_normal(mps, F)
        cur, Qrs = limit_sweep(mps, F)
        B = Bonds(mps, cur, Qrs)
        S = orth_basis(iota_matrix(cur, B.N - 1))
        defect = g - proj(S) @ g                              # (I - P_Shat) F Psi^0
        eta0 = norm(defect)
        scale = max(norm(g), 1e-30)
        # telescoping:  I - P_Shat = sum_k E_k
        w_id = max(w_id, norm(defect - sum(B.E(g, k) for k in range(B.N - 1))) / scale)
        # (I - P_Shat) G = D G = D R_0
        w_DR = max(w_DR, norm(defect - B.D(R0)) / scale, norm(defect - B.D(g)) / scale)
        # D is an orthogonal projector annihilating T_Psi M
        if it < 40:
            Dm = np.column_stack([B.D(e) for e in np.eye(len(psi0))])
            w_proj = max(w_proj, norm(Dm @ Dm - Dm), norm(Dm - Dm.conj().T))
            w_DT = max(w_DT, norm(Dm @ Tb))
        if eps > 1e-10:
            ratio = max(ratio, eta0 / eps)
            nviol += eta0 > eps * (1 + 1e-10)
        # nearest-neighbour generator: bond-k defect = two-site residual at bond k
        if local:
            ts = np.array([B.two_site_residual(g, k) for k in range(B.N - 1)])
            bk = np.array([B.bond_defect(R0, k) for k in range(B.N - 1)])
            w_ts = max(w_ts, norm(ts - bk) / scale)
            n_local += 1
        n += 1
    report("T13a telescoping  I - P_Shat = sum_k E_k", w_id < 1e-10, f"max = {w_id:.1e}")
    report("T13b (I - P_Shat) F Psi^0 = D F Psi^0 = D R_0   (dt -> 0)", w_DR < 1e-10,
           f"{n} instances, max rel. = {w_DR:.1e}")
    report("T13c D is an orthogonal projector with D T_Psi M = 0",
           max(w_proj, w_DT) < 1e-10,
           f"|D^2-D|, |D-D^*| <= {w_proj:.1e},  |D T| <= {w_DT:.1e}")
    report("T13d hence eta_0 <= eps  (constant exactly 1, every N)", nviol == 0,
           f"max eta_0/eps = {ratio:.4f}, {nviol} violations")
    report("T13e nearest-neighbour F: bond-k defect = two-site residual at bond k",
           w_ts < 1e-10, f"{n_local} instances, max rel. = {w_ts:.1e}")


# --------------------------------------------------------------------------- #
# T14  finite step size
# --------------------------------------------------------------------------- #


def t14_finite_dt(rng):
    w_split, worst_slack, n = 0.0, np.inf, 0
    cmax = 0.0
    for it in range(60):
        mps, F, ds, _ = random_instance(rng, Dmax=243)
        psi0, Tb, g, R0, eps = eps_and_normal(mps, F)
        nF = norm(F, 2)
        N = len(ds)
        for dt in (1.0 / nF, 1e-1 / nF, 1e-2 / nF, 1e-3 / nF):
            cur, Qrs = finite_sweep(mps, F, dt)
            B = Bonds(mps, cur, Qrs)
            S = orth_basis(iota_matrix(cur, N - 1))
            eta = norm(g - proj(S) @ g)
            lhs = norm(B.D(R0)) ** 2 + sum(norm(B.E(B.PP(g, k), k)) ** 2
                                          for k in range(N - 1))
            w_split = max(w_split, abs(eta ** 2 - lhs) / max(norm(g) ** 2, 1e-30))
            bound = eps + np.sqrt(N - 1) * 0.5 * dt * nF ** 2 * norm(psi0) * np.exp(dt * nF)
            worst_slack = min(worst_slack, (bound - eta) / max(norm(g), 1e-30))
            cmax = max(cmax, max(eta - eps, 0.0) / (dt * nF ** 2 * norm(psi0)))
            n += 1
    report("T14a eta^2 = |D R_0|^2 + sum_k |E_k P_k^*P_k F Psi^0|^2  (any dt)",
           w_split < 1e-10, f"{n} (instance, dt) pairs, max rel. = {w_split:.1e}")
    report("T14b eta <= eps + sqrt(N-1) (dt/2) |F|^2 |Psi^0| e^{dt|F|}",
           worst_slack >= -1e-12,
           f"min slack / |F Psi^0| = {worst_slack:.2e}; observed "
           f"max (eta-eps)_+/(dt |F|^2 |Psi^0|) = {cmax:.3f}")


# --------------------------------------------------------------------------- #
# T15  augmenting each bond with its two-site residual removes the defect
# --------------------------------------------------------------------------- #


def residual_sweep(mps, F, tol=1e-10):
    """First-order sweep in which bond k is augmented, in addition, with the column
    space of the two-site residual (I - Pi_k) Mat_k(iota_{k,k+1}^* F Psi^0)."""
    N = len(mps)
    ds = [A.shape[2] for A in mps]
    g = F @ mps_to_vector(mps)
    cur = [A.copy() for A in mps]
    for k in range(N - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        XK = (Ik.conj().T @ g).reshape(rl * d, rr)
        Lam = np.kron(left_blocks(cur, k), np.eye(d))
        Z = g.reshape(int(np.prod(ds[: k + 1])), -1)
        P1 = right_blocks(mps, k + 1) if k + 1 < N - 1 else np.ones((1, 1))
        Z3 = Z.reshape(Z.shape[0], ds[k + 1], -1) @ P1.conj().T
        two_site = Lam.conj().T @ Z3.reshape(Z.shape[0], -1)
        Q, U = qr(XC, mode="reduced")
        Qr = orth_basis(np.hstack([Q, XK, two_site]), tol=tol)
        T = Qr.conj().T @ Q
        cur[k] = ten_R(Qr, rl, d)
        Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
        cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
    S = orth_basis(iota_matrix(cur, N - 1))
    return norm(g - proj(S) @ g)


def t15_two_site_augmentation(rng):
    """For a dense generator the residual augmentation removes the defect only when
    the extra columns happen to fill the bond spaces (small chains); in general a
    defect remains, so the nearest-neighbour hypothesis is needed."""
    worst_nn, n_dense, n_left, ratios = 0.0, 0, 0, []
    for it in range(60):
        local = it % 2 == 0
        mps, F, ds, _ = random_instance(rng, Nmin=3, Dmax=243, local=local)
        _, _, g, _, eps = eps_and_normal(mps, F)
        eta2 = residual_sweep(mps, F) / max(norm(g), 1e-30)
        if local:
            worst_nn = max(worst_nn, eta2)
        elif eps > 1e-8:
            cur, Qrs = limit_sweep(mps, F)
            S = orth_basis(iota_matrix(cur, len(ds) - 1))
            eta0 = norm(g - proj(S) @ g) / norm(g)
            if eta0 > 1e-8:
                n_dense += 1
                ratios.append(eta2 / eta0)
                n_left += eta2 / eta0 > 1e-6
    report("T15a nearest-neighbour F: two-site residual augmentation gives eta_0 = 0",
           worst_nn < 1e-10, f"30 instances, max eta_0/|F Psi^0| = {worst_nn:.1e}")
    report("T15b control: for a dense F a defect generally remains",
           n_left > n_dense // 2,
           f"defect remains in {n_left}/{n_dense} dense instances; "
           f"eta_0(augmented)/eta_0 in [{min(ratios):.1e}, {max(ratios):.2f}]")


# --------------------------------------------------------------------------- #
# T16  the loss at each bond is part of the corner block at that cut
# --------------------------------------------------------------------------- #


def corner(psi0, g, ds, k, tol=1e-10):
    """|| (I - Pi^c_k) G^<k> (I - Pi^r_k) ||: the normal block of the matrix picture
    at cut k, with Pi^c, Pi^r the column and row projections of Psi^0 at that cut."""
    m = int(np.prod(ds[: k + 1]))
    U, s, Vh = np.linalg.svd(psi0.reshape(m, -1), full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    U, Vh = U[:, :r], Vh[:r]
    Gk = g.reshape(m, -1)
    Gk = Gk - U @ (U.conj().T @ Gk)
    return norm(Gk - (Gk @ Vh.conj().T) @ Vh)


def t16_loss_below_corner(rng):
    worst, n = -np.inf, 0
    for it in range(200):
        mps, F, ds, _ = random_instance(rng)
        psi0, _, g, _, _ = eps_and_normal(mps, F)
        cur, Qrs = limit_sweep(mps, F)
        B = Bonds(mps, cur, Qrs)
        for k in range(B.N - 1):
            c = corner(psi0, g, ds, k)
            worst = max(worst, (B.bond_defect(g, k) - c) / max(norm(g), 1e-30))
            n += 1
    report("T16  loss at bond k <= corner block of the field at cut k", worst < 1e-12,
           f"{n} bonds in 200 instances, max (loss - corner)/|F Psi^0| = {worst:.1e}")


def main():
    rng = np.random.default_rng(20261010)
    print("=" * 86)
    t12_bond_splitting(rng)
    t13_limit_structure(rng)
    t14_finite_dt(rng)
    t15_two_site_augmentation(rng)
    t16_loss_below_corner(rng)
    print("=" * 86)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - nfail} / {len(RESULTS)} checks passed")
    return nfail


if __name__ == "__main__":
    sys.exit(main())
