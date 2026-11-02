"""
Numerical checks for the claims of the note, and the shared dense machinery that
every other script in this folder imports.

NOTE: the labels below (conj:eta_eps, eq:no_tangent, rem:floor, prop:defect, ...)
refer to an early version of the note, which is no longer kept in this folder; they
do not match the labels of the current manuscript. The current statements are checked
by robustness_checks.py, order_checks.py and eta0_crosscheck.py, which import the
machinery of this file. The checks below remain valid as tests of that machinery.

The script works with *dense* objects on the full Hilbert space
    H = C^{d_0} (x) ... (x) C^{d_{N-1}},
so every map used in the note (the inclusion maps iota_k and iota_{k,k+1}, the
effective generators M_eff^[k] = iota_k^* F iota_k, the tangent-space projector of
the fixed-TT-rank manifold, the augmented Galerkin subspace Shat_{N-1}) is built
explicitly as a matrix.  Chains are therefore kept small (dim H <= a few hundred).
That is deliberate: the aim is to test the *mathematics* of the note, not to
provide an efficient implementation.

Conventions follow the note:

  * site tensor A^[k] has shape (r_{k-1}, r_k, d_k);
  * Mat_R(A) in C^{(r_{k-1} d_k) x r_k}  merges (alpha_{k-1}, p_k) lexicographically;
  * Mat_L(A) in C^{r_{k-1} x (r_k d_k)}  merges (alpha_k, p_k) lexicographically;
  * Psi^<k> is the matricisation of Psi across the cut at bond k, rows (p_0..p_k);
  * Lambda_k in C^{(d_0..d_{k-1}) x r_{k-1}} matricises L^[0]...L^[k-1];
  * P_k      in C^{r_k x (d_{k+1}..d_{N-1})} matricises R^[k+1]...R^[N-1];
  * iota_k(C)^<k> = (Lambda_k (x) I_{d_k}) Mat_R(C) P_k.

Tests (see main()):

  T1  lem:iota_iso           iota^* iota = Id for the one- and two-site maps.
  T2  prop:state_pres        the K-sweep leaves the represented state unchanged.
  T3  prop:exact             exactness on a TT-rank-r trajectory; the rank bound
                             rhat_k <= r_k + rho_k; the claim that
                             V_W^* P_k^* = Theta_k up to unitaries; and the two
                             load-bearing identities of the proof, K(t_1) =
                             Pbar_k Psi(t_1) and the equality of left supports,
                             together with the exclusion of rho_k > r_k.
  T4  prop:exact (sharpness) the overlap condition cannot be dropped.
  T5  prop:defect            iota_{k+1}(C) = iota_{k,k+1}(Ten_R(Q_c).C), its
                             adjoint, the zero block of C^[k+1], and
                             eq:two_site_residual to O(dt^2).
  T6  prop:local_defect(iii) eta <= eps + C' dt for N = 2.
  T7  eq:no_tangent          dimension count for T_Psi M versus Shat_{N-1}.
  T8  conj:eta_eps           (a) the dt -> 0 limit eta_0 <= eps over 200 random
                             instances; (b) a deliberate construction with eps = 0
                             and eta = Theta(dt) at N = 2, which refutes the
                             conjecture as literally stated; (c) the regime eps -> 0.
  T9  rem:floor              the error floor: one-step error / dt does not vanish.
  T10 eq:trunc_bound         sigma_1(Sigma_j) <= ||Psihat^1|| at every stage of the
                             backward sweep, and the bound itself.
  T11 alg:basisupdate K2     well-posedness of the thin QR and the full-column-rank
                             hypothesis behind eq:Vk0.

Run:  python benchmarks/bug_mps_checks.py
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import qr, svd, norm, matrix_rank
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


def expm_apply(A, v):
    """exp(A) @ v for a small dense A, without forming exp(A).

    The local solves only ever need the action of the exponential on the current
    coefficient vector.  expm_multiply is one to two orders of magnitude faster than
    forming expm(A) at these sizes and agrees with it to machine precision."""
    if A.shape[0] < 4:
        return expm(A) @ v
    return expm_multiply(A, v)

# --------------------------------------------------------------------------- #
# matricisations
# --------------------------------------------------------------------------- #


def mat_R(A):
    """A : (rl, rr, d) -> (rl*d, rr), rows lexicographic in (alpha_{k-1}, p_k)."""
    rl, rr, d = A.shape
    return A.transpose(0, 2, 1).reshape(rl * d, rr)


def ten_R(X, rl, d):
    rr = X.shape[1]
    return X.reshape(rl, d, rr).transpose(0, 2, 1)


def mat_L(A):
    """A : (rl, rr, d) -> (rl, rr*d), cols lexicographic in (alpha_k, p_k)."""
    rl, rr, d = A.shape
    return A.reshape(rl, rr * d)


def ten_L(X, rr, d):
    rl = X.shape[0]
    return X.reshape(rl, rr, d)


# --------------------------------------------------------------------------- #
# MPS utilities
# --------------------------------------------------------------------------- #


def mps_to_vector(mps):
    out = np.ones((1, 1), dtype=complex)
    for A in mps:
        rl, rr, d = A.shape
        out = out.reshape(-1, rl) @ A.transpose(0, 2, 1).reshape(rl, d * rr)
        out = out.reshape(-1, rr)
    return out.reshape(-1)


def random_tensor(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def feasible_ranks(ds, rs):
    """Clip requested bond dimensions to those a chain with locals ds can attain."""
    out = [int(x) for x in rs]
    N = len(ds)
    left = 1
    for k in range(N - 1):
        left = left * ds[k]
        out[k] = min(out[k], left)
        left = out[k]
    right = 1
    for k in range(N - 2, -1, -1):
        right = right * ds[k + 1]
        out[k] = min(out[k], right)
        right = out[k]
    return out


def random_mps(rng, ds, rs):
    """Random MPS of exact TT-rank rs, mixed-canonical with OC at site 0."""
    N = len(ds)
    rs = feasible_ranks(ds, rs)
    bonds = [1] + list(rs) + [1]
    mps = [random_tensor(rng, (bonds[k], bonds[k + 1], ds[k])) for k in range(N)]
    return right_canonicalise(mps)


def right_canonicalise(mps):
    """Sites 1..N-1 right-isometric, OC at site 0."""
    cur = [A.copy() for A in mps]
    N = len(cur)
    for j in range(N - 1, 0, -1):
        rl, rr, d = cur[j].shape
        M = mat_L(cur[j])
        Q, R = qr(M.conj().T, mode="reduced")          # M^* = Q R  =>  M = R^* Q^*
        s = Q.shape[1]
        cur[j] = ten_L(Q.conj().T, rr, d)              # (s, rr, d)
        cur[j - 1] = np.einsum("abp,bc->acp", cur[j - 1], R.conj().T)
    return cur


def left_blocks(mps, k):
    """Lambda_k in C^{(d_0..d_{k-1}) x r_{k-1}}."""
    out = np.ones((1, 1), dtype=complex)
    for j in range(k):
        rl, rr, d = mps[j].shape
        out = out.reshape(-1, rl) @ mps[j].transpose(0, 2, 1).reshape(rl, d * rr)
        out = out.reshape(-1, rr)
    return out


def right_blocks(mps, k):
    """P_k in C^{r_k x (d_{k+1}..d_{N-1})}."""
    N = len(mps)
    out = np.ones((1, 1), dtype=complex)               # (r_{N-1}, 1)
    for j in range(N - 1, k, -1):
        rl, rr, d = mps[j].shape
        T = mps[j].transpose(0, 2, 1).reshape(rl * d, rr)      # ((rl,p), rr)
        out = (T @ out).reshape(rl, d * out.shape[1])
    return out


def iota_matrix(mps, k):
    """Dense iota_k acting on the row-major vec of Mat_R(C)."""
    Lam = left_blocks(mps, k)
    P = right_blocks(mps, k)
    d = mps[k].shape[2]
    A = np.kron(Lam, np.eye(d))
    return np.kron(A, P.T)


def iota2_matrix(mps, k):
    """Dense iota_{k,k+1} acting on the row-major vec of Mat_k(D)."""
    Lam = left_blocks(mps, k)
    P = right_blocks(mps, k + 1)
    dk = mps[k].shape[2]
    dk1 = mps[k + 1].shape[2]
    A = np.kron(Lam, np.eye(dk))
    B = np.kron(np.eye(dk1), P)
    return np.kron(A, B.T)


def orth_basis(M, tol=1e-10):
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0), dtype=complex)
    U, s, _ = svd(M, full_matrices=False)
    r = int(np.sum(s > tol * max(1.0, float(s[0]))))
    return U[:, :r]


def proj(B):
    return B @ B.conj().T


# --------------------------------------------------------------------------- #
# BUG-MPS step
# --------------------------------------------------------------------------- #


def bug_mps_step(mps, F, dt, theta=0.0, exact_rhs=None, complete_qr=False,
                 tol=1e-10, rmax=None):
    """One BUG-MPS step (alg:timestep) with exact local solves.

    F          dense generator on H for dPsi/dt = F Psi (time independent), or None.
    exact_rhs  pair (Psi(t_0), Psi(t_1)) for the state-independent right-hand side
               F(t,.) = dPsi(t)/dt, for which K^[k](t_1) = C^[k] + iota_k^*(Psi(t_1)
               - Psi(t_0)) exactly.
    complete_qr  keep the full unitary Q_c at every bond (rem:uncapped_exact).
    rmax       optional hard cap on the bond dimension in the compression sweep.
    """
    N = len(mps)
    cur = [A.copy() for A in mps]
    rec = {"rhats": [], "Qr": {}, "K": {}, "T": {}}

    for k in range(N - 1):                               # K-step sweep
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        assert rl * d >= rr, f"thin QR at bond {k} needs rhat_(k-1) d_k >= r_k"

        if exact_rhs is not None:
            psi0, psi1 = exact_rhs
            XK = XC + (Ik.conj().T @ (psi1 - psi0)).reshape(rl * d, rr)
        else:
            Meff = Ik.conj().T @ F @ Ik
            XK = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)

        Q, U = qr(XC, mode="reduced")                    # K2
        if complete_qr:                                  # K3
            Qc, _ = qr(np.hstack([orth_basis(np.hstack([Q, XK]), tol),
                                  np.eye(rl * d)]), mode="reduced")
            Qr = Qc[:, : rl * d]
        else:
            Qr = orth_basis(np.hstack([Q, XK]), tol=tol)
        rec["rhats"].append(Qr.shape[1])

        T = Qr.conj().T @ Q                              # K4
        cur[k] = ten_R(Qr, rl, d)
        Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
        cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
        rec["Qr"][k], rec["K"][k], rec["T"][k] = Qr, XK, T

    k = N - 1                                            # S-step
    rl, rr, d = cur[k].shape
    Ik = iota_matrix(cur, k)
    rec["iota_last"] = Ik
    XC = mat_R(cur[k])
    if exact_rhs is not None:
        psi0, psi1 = exact_rhs
        XC1 = XC + (Ik.conj().T @ (psi1 - psi0)).reshape(rl * d, rr)
    else:
        Meff = Ik.conj().T @ F @ Ik
        XC1 = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)
    cur[k] = ten_R(XC1, rl, d)
    rec["pre_trunc"] = [A.copy() for A in cur]

    cur, trec = backward_svd_sweep(cur, theta, rmax)     # truncation
    rec["trunc"] = trec
    return cur, rec


def backward_svd_sweep(mps, theta, rmax=None):
    """alg:trunc with a relative per-bond tolerance theta (and an optional cap rmax)."""
    cur = [A.copy() for A in mps]
    N = len(cur)
    rec = {"sigma1": [], "discarded": [], "kept": []}
    for j in range(N - 1, 0, -1):
        rl, rr, d = cur[j].shape
        U, s, Vh = svd(mat_L(cur[j]), full_matrices=False)
        s1 = float(s[0]) if s.size else 0.0
        keep = max(int(np.sum(s >= theta * s1)), 1)
        if rmax is not None:
            keep = min(keep, rmax)
        rec["sigma1"].append(s1)
        rec["discarded"].append(s[keep:].copy())
        rec["kept"].append(keep)
        U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
        cur[j] = ten_L(Vh, rr, d)
        cur[j - 1] = np.einsum("abp,bc->acp", cur[j - 1], U * s)
    return cur, rec


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def tangent_basis(mps):
    """Orthonormal basis of T_Psi M_r = sum_k iota_k(C^{r_{k-1} x r_k x d_k})."""
    N = len(mps)
    cols = []
    cur = [A.copy() for A in mps]
    for k in range(N):
        cols.append(iota_matrix(cur, k))
        if k < N - 1:
            rl, rr, d = cur[k].shape
            Q, U = qr(mat_R(cur[k]), mode="reduced")
            cur[k] = ten_R(Q, rl, d)
            cur[k + 1] = np.einsum("ab,bcp->acp", U, cur[k + 1])
    return orth_basis(np.hstack(cols))


def eta_eps(mps, F, dt, tol=1e-10):
    """eq:eta_eps for a linear generator F at the initial state."""
    psi0 = mps_to_vector(mps)
    _, rec = bug_mps_step(mps, F, dt, theta=0.0, tol=tol)
    S = orth_basis(rec["iota_last"])
    Tb = tangent_basis(mps)
    v = F @ psi0
    return norm(v - proj(S) @ v), norm(v - proj(Tb) @ v), rec


def eta_limit(mps, F, tol=1e-10):
    """eta in the limit dt -> 0: at every bond augment Mat_R(C^[k]) with the
    first-order direction Mat_R(iota_k^* F iota_k(C^[k])) instead of with
    Mat_R(K^[k](t_1)).  This is the dt -> 0 limit of Vhat_k, hence of eta."""
    N = len(mps)
    cur = [A.copy() for A in mps]
    for k in range(N - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        XK = ((Ik.conj().T @ F @ Ik) @ XC.reshape(-1)).reshape(rl * d, rr)
        Q, U = qr(XC, mode="reduced")
        Qr = orth_basis(np.hstack([Q, XK]), tol=tol)
        T = Qr.conj().T @ Q
        cur[k] = ten_R(Qr, rl, d)
        Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
        cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
    S = orth_basis(iota_matrix(cur, N - 1))
    v = F @ mps_to_vector(mps)
    return norm(v - proj(S) @ v)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


def t1_iota_isometry(rng):
    worst = 0.0
    for _ in range(20):
        N = int(rng.integers(2, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 4, size=N - 1)]
        cur = random_mps(rng, ds, rs)
        for k in range(N):
            I = iota_matrix(cur, k)
            worst = max(worst, norm(I.conj().T @ I - np.eye(I.shape[1])))
            if k < N - 1:
                I2 = iota2_matrix(cur, k)
                worst = max(worst, norm(I2.conj().T @ I2 - np.eye(I2.shape[1])))
                rl, rr, d = cur[k].shape
                Q, U = qr(mat_R(cur[k]), mode="reduced")
                cur[k] = ten_R(Q, rl, d)
                cur[k + 1] = np.einsum("ab,bcp->acp", U, cur[k + 1])
    report("T1  lem:iota_iso   iota^* iota = Id (one- and two-site)",
           worst < 1e-10, f"max ||iota^* iota - I|| = {worst:.2e}")


def t2_state_preservation(rng):
    worst = 0.0
    for _ in range(15):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 4, size=N - 1)]
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D))
        psi0 = mps_to_vector(mps)
        cur = [A.copy() for A in mps]
        for k in range(N - 1):
            rl, rr, d = cur[k].shape
            Ik = iota_matrix(cur, k)
            XC = mat_R(cur[k])
            XK = (expm(0.1 * (Ik.conj().T @ F @ Ik)) @ XC.reshape(-1)
                  ).reshape(rl * d, rr)
            Q, U = qr(XC, mode="reduced")
            Qr = orth_basis(np.hstack([Q, XK]))
            T = Qr.conj().T @ Q
            cur[k] = ten_R(Qr, rl, d)
            Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
            cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
            worst = max(worst, norm(mps_to_vector(cur) - psi0) / norm(psi0))
    report("T2  prop:state_pres  Psi^(k) = Psi^0 along the sweep",
           worst < 1e-10, f"max relative drift = {worst:.2e}")


def rank_r_curve(rng, ds, rs, dt):
    rs = feasible_ranks(ds, rs)
    N = len(ds)
    bonds = [1] + list(rs) + [1]
    A0 = [random_tensor(rng, (bonds[k], bonds[k + 1], ds[k])) for k in range(N)]
    Ad = [random_tensor(rng, (bonds[k], bonds[k + 1], ds[k])) for k in range(N)]
    mps0 = right_canonicalise(A0)
    mps1 = [A0[k] + dt * Ad[k] for k in range(N)]
    return mps0, mps1


def schmidt(psi, ds, k, tol=1e-10):
    dl = int(np.prod(ds[: k + 1]))
    M = psi.reshape(dl, -1)
    U, s, Vh = svd(M, full_matrices=False)
    r = int(np.sum(s > tol * s[0]))
    return U[:, :r], s[:r], Vh[:r].conj().T, r      # V has orthonormal columns


def t3_exactness(rng):
    worst_err, ok_rank, trials = 0.0, True, 0
    for _ in range(40):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 3, size=N - 1)]
        dt = 0.3
        mps0, mps1 = rank_r_curve(rng, ds, rs, dt)
        psi0, psi1 = mps_to_vector(mps0), mps_to_vector(mps1)
        rsf = feasible_ranks(ds, rs)
        rhos, nondeg = [], True
        for k in range(N - 1):
            _, _, V1, rho = schmidt(psi1, ds, k)
            _, _, V0, r0 = schmidt(psi0, ds, k)
            rhos.append(rho)
            if matrix_rank(V1.conj().T @ V0, tol=1e-9) != rho:
                nondeg = False
        if not nondeg:
            continue
        trials += 1
        out, rec = bug_mps_step(mps0, None, dt, theta=1e-9, exact_rhs=(psi0, psi1))
        worst_err = max(worst_err, norm(mps_to_vector(out) - psi1) / norm(psi1))
        for k in range(N - 1):
            if rec["rhats"][k] > rsf[k] + rhos[k]:
                ok_rank = False
    report("T3a prop:exact  Psi^1 = Psi(t_1)", worst_err < 1e-9,
           f"{trials} trials, max relative error = {worst_err:.2e}")
    report("T3b prop:exact  rhat_k <= r_k + rho_k", ok_rank)


def t3c_theta_identity(rng):
    worst = 0.0
    for _ in range(25):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 3, size=N - 1)]
        dt = 0.3
        mps0, mps1 = rank_r_curve(rng, ds, rs, dt)
        psi0, psi1 = mps_to_vector(mps0), mps_to_vector(mps1)
        cur = [A.copy() for A in mps0]
        for k in range(N - 1):
            rl, rr, d = cur[k].shape
            Lam, P = left_blocks(cur, k), right_blocks(cur, k)
            dl = int(np.prod(ds[: k + 1]))
            Wk = np.kron(Lam, np.eye(d)).conj().T @ psi1.reshape(dl, -1)
            _, sW, VWh = svd(Wk, full_matrices=False)
            rho = int(np.sum(sW > 1e-10 * sW[0]))
            VW = VWh[:rho].conj().T
            lhs = VW.conj().T @ P.conj().T
            _, _, V1, _ = schmidt(psi1, ds, k)
            _, _, V0, _ = schmidt(psi0, ds, k)
            Theta = V1.conj().T @ V0
            a = np.sort(svd(lhs, compute_uv=False))
            b = np.sort(svd(Theta, compute_uv=False))
            m = min(a.size, b.size)
            worst = max(worst, float(norm(a[-m:] - b[-m:])))
            Ik = iota_matrix(cur, k)
            XC = mat_R(cur[k])
            XK = XC + (Ik.conj().T @ (psi1 - psi0)).reshape(rl * d, rr)
            Q, U = qr(XC, mode="reduced")
            Qr = orth_basis(np.hstack([Q, XK]))
            T = Qr.conj().T @ Q
            cur[k] = ten_R(Qr, rl, d)
            Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
            cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
    report("T3c prop:exact  V_W^* P_k^* = Theta_k up to unitaries",
           worst < 1e-9, f"max singular-value mismatch = {worst:.2e}")


def t3d_proof_steps(rng):
    """The two load-bearing identities of the proof of prop:exact.

    Step 1   |K(t_1)> = Pbar_i |A(t_1)>  at every bond, exactly.
    Step 2   left supp Pbar_i|A(t_1)> = left supp |A(t_1)>  under eq:overlap.
    Step 2'  eq:overlap forbids rho_i > chi_i (so rhat_i <= 2 chi_i).
    """
    worst_K, ok_supp, trials = 0.0, True, 0
    for _ in range(40):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 3, size=N - 1)]
        mps0, mps1 = rank_r_curve(rng, ds, rs, 0.3)
        psi0, psi1 = mps_to_vector(mps0), mps_to_vector(mps1)

        hyp = True                                        # eq:overlap at every bond
        for k in range(N - 1):
            dl = int(np.prod(ds[: k + 1]))
            _, _, V0, _ = schmidt(psi0, ds, k)
            M1 = psi1.reshape(dl, -1)
            if matrix_rank(M1 @ V0 @ V0.conj().T, tol=1e-9) != matrix_rank(M1, tol=1e-9):
                hyp = False
        if not hyp:
            continue
        trials += 1

        cur = [A.copy() for A in mps0]
        for k in range(N - 1):
            rl, rr, d = cur[k].shape
            dl = int(np.prod(ds[: k + 1]))
            Ik = iota_matrix(cur, k)
            XC = mat_R(cur[k])
            XK = XC + (Ik.conj().T @ (psi1 - psi0)).reshape(rl * d, rr)

            _, _, V0, _ = schmidt(psi0, ds, k)
            PbarA = psi1.reshape(dl, -1) @ V0 @ V0.conj().T
            worst_K = max(worst_K, norm(Ik @ XK.reshape(-1) - PbarA.reshape(-1))
                          / norm(psi1))

            QA = orth_basis(PbarA, tol=1e-10)             # the two left supports
            QB = orth_basis(psi1.reshape(dl, -1), tol=1e-10)
            if QA.shape[1] != QB.shape[1] or norm(QA @ QA.conj().T
                                                  - QB @ QB.conj().T) > 1e-9:
                ok_supp = False

            Q, U = qr(XC, mode="reduced")
            Qr = orth_basis(np.hstack([Q, XK]), tol=1e-10)
            T = Qr.conj().T @ Q
            cur[k] = ten_R(Qr, rl, d)
            Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
            cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)

    report("T3d prop:exact Step 1  K(t_1) = Pbar_k Psi(t_1)", worst_K < 1e-12,
           f"{trials} trials, max relative deviation = {worst_K:.2e}")
    report("T3e prop:exact Step 2  the two left supports coincide", ok_supp)

    # Step 2': a target whose rank at the middle bond exceeds chi must fail eq:overlap
    ds, dl, excluded = [2, 2, 2, 2], 4, True
    for chi, rho in ((1, 2), (1, 3), (2, 3), (2, 4), (3, 4)):
        U0, s0, V0h = svd(random_tensor(rng, (dl, dl)))
        psi0 = (U0[:, :chi] * s0[:chi]) @ V0h[:chi]
        U1, s1, V1h = svd(random_tensor(rng, (dl, dl)))
        psi1 = (U1[:, :rho] * s1[:rho]) @ V1h[:rho]
        _, _, V0, _ = schmidt(psi0.reshape(-1), ds, 1)
        if matrix_rank(psi1 @ V0 @ V0.conj().T, tol=1e-9) == matrix_rank(psi1, tol=1e-9):
            excluded = False
    report("T3f prop:exact  eq:overlap excludes rho_k > r_k", excluded,
           "rank(Pbar_k Psi(t_1)) <= r_k, so the hypothesis cannot hold")


def t4_degenerate(rng):
    N, d = 3, 4
    errs = []
    for _ in range(10):
        bonds = [1, 2, 2, 1]
        A0 = [random_tensor(rng, (bonds[k], bonds[k + 1], d)) for k in range(N)]
        A1 = [random_tensor(rng, (bonds[k], bonds[k + 1], d)) for k in range(N)]
        A0[2][:, :, 2:] = 0.0
        A1[2][:, :, :2] = 0.0
        mps0 = right_canonicalise(A0)
        psi0, psi1 = mps_to_vector(mps0), mps_to_vector(A1)
        out, _ = bug_mps_step(mps0, None, 1.0, theta=1e-12, exact_rhs=(psi0, psi1))
        errs.append(norm(mps_to_vector(out) - psi1) / norm(psi1))
    report("T4  prop:exact  Theta_1 = 0 gives an O(1) one-step error",
           min(errs) > 0.1, f"relative errors in [{min(errs):.3f}, {max(errs):.3f}]")


def t5_two_site_residual(rng):
    worst_id = worst_adj = worst_zero = 0.0
    ratios = []
    for _ in range(8):
        N = int(rng.integers(3, 5))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 3, size=N - 1)]
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D))
        psi0 = mps_to_vector(mps)
        errs = {}
        for dt in (1e-2, 5e-3):
            cur = [A.copy() for A in mps]
            k = 0
            rl, rr, d = cur[k].shape
            Ik = iota_matrix(cur, k)
            XC = mat_R(cur[k])
            XK = (expm(dt * (Ik.conj().T @ F @ Ik)) @ XC.reshape(-1)
                  ).reshape(rl * d, rr)
            Q, U = qr(XC, mode="reduced")
            Qr_thin = orth_basis(np.hstack([Q, XK]))
            rhat = Qr_thin.shape[1]
            Qc, _ = qr(np.hstack([Qr_thin, np.eye(rl * d)]), mode="reduced")
            Qc = Qc[:, : rl * d]
            Qperp = Qc[:, rhat:]
            Pi_k = Qr_thin @ Qr_thin.conj().T
            I2 = iota2_matrix(cur, k)

            cur[k] = ten_R(Qc, rl, d)
            T = Qc.conj().T @ Q
            Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
            cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
            rl1, rr1, d1 = cur[k + 1].shape

            worst_zero = max(worst_zero,
                             norm(cur[k + 1][rhat:]) / max(norm(cur[k + 1]), 1e-30))

            # iota_{k+1}(C) = iota_{k,k+1}(Ten_R(Q_c).C)
            Ik1 = iota_matrix(cur, k + 1)
            lhs = Ik1 @ mat_R(cur[k + 1]).reshape(-1)
            D2 = np.einsum("ab,bcp->acp", Qc, cur[k + 1])          # (rl*d, rr1, d1)
            Matk = D2.transpose(0, 2, 1).reshape(rl * d, d1 * rr1)
            worst_id = max(worst_id, norm(lhs - I2 @ Matk.reshape(-1)) / norm(lhs))

            # adjoint relation
            Phi = random_tensor(rng, (D,))
            a = (Ik1.conj().T @ Phi).reshape(rl1 * d1, rr1)
            b = (Qc.conj().T @ (I2.conj().T @ Phi).reshape(rl * d, d1 * rr1))
            b = b.reshape(rl1, d1, rr1).reshape(rl1 * d1, rr1)
            worst_adj = max(worst_adj, norm(a - b) / max(norm(a), 1e-30))

            # eq:two_site_residual
            XC1 = mat_R(cur[k + 1])
            XK1 = (expm(dt * (Ik1.conj().T @ F @ Ik1)) @ XC1.reshape(-1)
                   ).reshape(rl1 * d1, rr1)
            Kt = XK1.reshape(rl1, d1, rr1)
            lhs_v = np.einsum("ab,bpc->apc", Qperp, Kt[rhat:]
                              ).reshape(rl * d, d1 * rr1)
            two_site = (I2.conj().T @ (F @ psi0)).reshape(rl * d, d1 * rr1)
            rhs_v = dt * (np.eye(rl * d) - Pi_k) @ two_site
            errs[dt] = norm(lhs_v - rhs_v)
        if errs[5e-3] > 1e-14:
            ratios.append(errs[1e-2] / errs[5e-3])
    report("T5a prop:defect  C^[k+1] has no V_k^perp block", worst_zero < 1e-12,
           f"max = {worst_zero:.2e}")
    report("T5b prop:defect  iota_{k+1}(C) = iota_{k,k+1}(Ten_R(Q_c).C)",
           worst_id < 1e-10, f"max = {worst_id:.2e}")
    report("T5c prop:defect  adjoint relation", worst_adj < 1e-10,
           f"max = {worst_adj:.2e}")
    ok = bool(ratios) and all(r > 2.5 for r in ratios)
    report("T5d prop:defect  eq:two_site_residual remainder is O(dt^2)", ok,
           f"error ratio on halving dt = {[round(r, 2) for r in ratios]} (expect ~4)")


def t6_N2(rng):
    """prop:local_defect(iii):  eta <= eps + C' dt.  Testing "eta <= eps" alone is
    wrong -- the dt term is genuinely needed, as t8b shows -- so we test instead
    that the excess (eta - eps)_+ is O(dt) with a dt-independent constant."""
    worst_excess, worst_const = 0.0, 0.0
    for _ in range(50):
        d0, d1 = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        r = int(rng.integers(1, min(d0, d1) + 1))
        mps = random_mps(rng, [d0, d1], [r])
        F = random_tensor(rng, (d0 * d1, d0 * d1))
        for dt in (1e-1, 1e-2, 1e-3):
            eta, eps, _ = eta_eps(mps, F, dt)
            excess = max(eta - eps, 0.0)
            worst_excess = max(worst_excess, excess)
            worst_const = max(worst_const, excess / (dt * norm(F)))
    report("T6  prop:local_defect(iii)  (eta - eps)_+ <= C' dt for N = 2",
           worst_const < 5.0,
           f"max (eta-eps)_+ = {worst_excess:.2e}; "
           f"max (eta-eps)_+/(dt ||F||) = {worst_const:.3f} (dt-independent)")


def t7_no_tangent(rng):
    rows, consistent = [], True
    for ds, rs in [([2, 2, 2, 2], [2, 2, 2]), ([3, 3, 3], [2, 2]),
                   ([2, 2, 2, 2, 2], [2, 2, 2, 2]), ([10, 2], [1]),
                   ([2, 2, 2], [2, 2])]:
        N = len(ds)
        rsf = feasible_ranks(ds, rs)
        bonds = [1] + rsf + [1]
        dimT = sum(bonds[k] * bonds[k + 1] * ds[k] for k in range(N)) \
            - sum(bonds[k + 1] ** 2 for k in range(N - 1))
        dimS_max = min(2 * rsf[-1], int(np.prod(ds[:-1]))) * ds[-1]
        mps = random_mps(rng, ds, rsf)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D))
        _, rec = bug_mps_step(mps, F, 0.1, theta=0.0)
        S = orth_basis(rec["iota_last"])
        Tb = tangent_basis(mps)
        contained = bool(norm(Tb - proj(S) @ Tb) < 1e-9)
        rows.append((ds, rsf, dimT, Tb.shape[1], dimS_max, S.shape[1], contained))
        if dimT > dimS_max and contained:
            consistent = False
    print("        d                r          dim T (formula/num)   dim Shat (max/actual)   T subset Shat")
    for ds, rs, dimT, dimTn, dS, dSn, c in rows:
        print(f"        {str(ds):16s} {str(rs):10s} {dimT:3d} / {dimTn:3d}"
              f"              {dS:3d} / {dSn:3d}             {c}")
    strict = [r for r in rows if r[2] > r[4]]
    report("T7  eq:no_tangent  T_Psi M is not inside Shat_{N-1} (dimension count)",
           consistent and len(strict) > 0,
           f"{len(strict)}/{len(rows)} configurations have dim T > max dim Shat, "
           f"including the N = 2 case d = (10,2), r = 1")


def two_local_generator(rng, ds):
    """A nearest-neighbour (two-local) non-Hermitian generator on H."""
    N = len(ds)
    D = int(np.prod(ds))
    F = np.zeros((D, D), dtype=complex)
    for j in range(N - 1):
        h = random_tensor(rng, (ds[j] * ds[j + 1], ds[j] * ds[j + 1]))
        left = int(np.prod(ds[:j])) if j else 1
        right = int(np.prod(ds[j + 2:])) if j + 2 < N else 1
        F += np.kron(np.kron(np.eye(left), h), np.eye(right))
    return F


def t8_conjecture(rng):
    """conj:eta_eps.  Two separate statements have to be distinguished:
    (a) the dt -> 0 limit eta_0 <= eps, which the numerics support, and
    (b) eta <= c eps at fixed dt > 0, which is false."""
    worst_ratio, nviol, ntot = 0.0, 0, 0
    while ntot < 200:
        N = int(rng.integers(2, 7))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        if int(np.prod(ds)) > 256:          # keep the dense objects small
            continue
        rs = [int(x) for x in rng.integers(1, 4, size=N - 1)]
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D)) if rng.random() < 0.5 \
            else two_local_generator(rng, ds)
        psi0 = mps_to_vector(mps)
        v = F @ psi0
        eps = norm(v - proj(tangent_basis(mps)) @ v)
        e0 = eta_limit(mps, F)
        ntot += 1
        if eps > 1e-10:
            worst_ratio = max(worst_ratio, e0 / eps)
            if e0 > eps * (1 + 1e-8):
                nviol += 1
    report("T8a conj:eta_eps in the dt -> 0 limit: eta_0 <= eps", nviol == 0,
           f"{ntot} random instances (dense and two-local generators), "
           f"max eta_0/eps = {worst_ratio:.4f}, {nviol} violations")

    # Counterexample to the conjecture AS STATED (c independent of the step size).
    # N = 2, d = (5,2), r = 2:  M_2 is open and dense in C^{5x2}, so T_Psi M is all
    # of C^{5x2} and eps = 0 identically; but dim Vhat_0 <= 2r = 4 < 5 = d_0, so
    # eta > 0 for every dt > 0, with eta = Theta(dt).
    rows = []
    for _ in range(4):
        mps = random_mps(rng, [5, 2], [2])
        F = random_tensor(rng, (10, 10))
        rows.append([(dt,) + eta_eps(mps, F, dt)[:2] for dt in (1e-1, 1e-2, 1e-3)]
                    + [("lim", eta_limit(mps, F), 0.0)])
    print("        counterexample N = 2, d = (5,2), r = 2  (M_2 open dense => eps = 0):")
    for data in rows:
        print("          " + "  ".join(
            f"{'dt='+format(dt,'g') if dt != 'lim' else 'dt->0'}: "
            f"eta={eta:.2e} eps={eps:.0e}" for dt, eta, eps in data))
    flat = [x for data in rows for x in data[:3]]
    ok = all(eps < 1e-12 for _, _, eps in flat) and all(eta > 1e-6 for _, eta, _ in flat)
    rat = [data[0][1] / data[1][1] for data in rows if data[1][1] > 1e-14]
    report("T8b conj:eta_eps as literally stated (eta <= c eps) is FALSE", ok,
           f"eps = 0 but eta > 0; eta(dt)/eta(dt/10) = "
           f"{[round(x, 1) for x in rat]} (~10, i.e. eta = Theta(dt))")

    # The regime the conjecture is actually needed in: eps -> 0 at fixed state.
    # F_lambda = F_0 - (1-lambda) * (normal part of F_0 Psi^0) Psi^{0*}/||Psi^0||^2
    # has eps(F_lambda) = lambda eps(F_0) and is otherwise generic.
    print("        eta_0/eps as eps -> 0 (lambda scales the normal part of F Psi^0):")
    ratios = []
    for ds, rs in [([3, 3, 3], [2, 2]), ([3, 3, 3, 3], [2, 2, 2]),
                   ([4, 3, 3], [2, 2])]:
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        psi0 = mps_to_vector(mps)
        Tb = tangent_basis(mps)
        F0 = random_tensor(rng, (D, D))
        normal = (F0 @ psi0) - proj(Tb) @ (F0 @ psi0)
        nrm = float(np.vdot(psi0, psi0).real)
        line = []
        for lam in (1.0, 1e-2, 1e-4):
            F = F0 - (1 - lam) * np.outer(normal, psi0.conj()) / nrm
            v = F @ psi0
            e = norm(v - proj(Tb) @ v)
            e0 = eta_limit(mps, F)
            line.append(f"lam={lam:.0e}: eta_0/eps={e0/e:.4f}")
            ratios.append(e0 / e)
        print(f"          d={str(ds):12s} " + "  ".join(line))
    report("T8c conj:eta_eps  eta_0/eps stays bounded as eps -> 0",
           max(ratios) <= 1.0 + 1e-8, f"max eta_0/eps = {max(ratios):.4f}")


def t9_error_floor(rng):
    # The local dimensions must not saturate the bonds: with d = 2 and r = 2 the
    # augmented bonds fill the whole bond space, Shat_{N-1} = H, and eta = 0.
    ds, rs = [3, 3, 3, 3], [2, 2, 2]
    mps = random_mps(rng, ds, rs)
    D = int(np.prod(ds))
    F = random_tensor(rng, (D, D))
    psi0 = mps_to_vector(mps)
    print("          dt        ||Psi^1 - Psi(t_1)||        err/dt           eta")
    data = []
    for dt in (1e-1, 1e-2, 1e-3, 1e-4):
        out, _ = bug_mps_step(mps, F, dt, theta=0.0)
        err = norm(mps_to_vector(out) - expm(dt * F) @ psi0)
        eta, _, _ = eta_eps(mps, F, dt)
        data.append((dt, err, err / dt, eta))
        print(f"          {dt:.0e}   {err:.6e}             {err/dt:.6e}   {eta:.6e}")
    ok = (abs(data[-1][2] - data[-2][2]) / max(data[-1][2], 1e-30) < 0.05 and
          abs(data[-1][2] - data[-1][3]) / data[-1][3] < 0.05)
    report("T9  rem:floor  one-step error / dt tends to a nonzero limit (= eta)", ok,
           f"err/dt = {data[-1][2]:.4e}, eta = {data[-1][3]:.4e}")


def t10_truncation(rng):
    worst_sigma, worst_slack = 0.0, np.inf
    for _ in range(20):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 4, size=N - 1)]
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D))
        theta = float(10 ** rng.uniform(-3, -1))
        out, rec = bug_mps_step(mps, F, 0.2, theta=theta)
        pre = mps_to_vector(rec["pre_trunc"])
        npre = norm(pre)
        for s1 in rec["trunc"]["sigma1"]:
            worst_sigma = max(worst_sigma, s1 / npre)
        err = norm(mps_to_vector(out) - pre)
        bound = np.sqrt((N - 1) * max(rec["rhats"] + [1])) * theta * npre
        worst_slack = min(worst_slack, bound / max(err, 1e-300))
    report("T10a eq:trunc_bound  sigma_1(Sigma_j) <= ||Psihat^1|| at every stage",
           worst_sigma <= 1 + 1e-12, f"max sigma_1/||Psihat^1|| = {worst_sigma:.6f}")
    report("T10b eq:trunc_bound  the bound holds", worst_slack >= 1.0,
           f"min bound/error = {worst_slack:.2f}")


def t11_qr_wellposed(rng):
    """alg:basisupdate K2.  The thin QR of Mat_R(C^[k]) in C^{(rhat_{k-1} d_k) x r_k}
    needs rhat_{k-1} d_k >= r_k, and eq:Vk0 needs Mat_R(C^[k]) to have full column
    rank.  Both follow from Psi^0 having ATTAINED bond dimensions (Psi^0 in M_r, not
    merely in M_{<=r}); neither is stated in the note."""
    ok_shape, ok_rank = True, True
    for _ in range(20):
        N = int(rng.integers(3, 6))
        ds = [int(x) for x in rng.integers(2, 4, size=N)]
        rs = [int(x) for x in rng.integers(1, 5, size=N - 1)]
        mps = random_mps(rng, ds, rs)
        D = int(np.prod(ds))
        F = random_tensor(rng, (D, D))
        cur = [A.copy() for A in mps]
        for k in range(N - 1):
            rl, rr, d = cur[k].shape
            if rl * d < rr:
                ok_shape = False
            if matrix_rank(mat_R(cur[k]), tol=1e-9) < rr:
                ok_rank = False
            Ik = iota_matrix(cur, k)
            XC = mat_R(cur[k])
            XK = (expm(0.1 * (Ik.conj().T @ F @ Ik)) @ XC.reshape(-1)
                  ).reshape(rl * d, rr)
            Q, U = qr(XC, mode="reduced")
            Qr = orth_basis(np.hstack([Q, XK]))
            T = Qr.conj().T @ Q
            cur[k] = ten_R(Qr, rl, d)
            Ctil = np.einsum("ab,bcp->acp", U, cur[k + 1])
            cur[k + 1] = np.einsum("ab,bcp->acp", T, Ctil)
    report("T11a alg:basisupdate K2  rhat_{k-1} d_k >= r_k at every bond", ok_shape)
    report("T11b eq:Vk0  Mat_R(C^[k]) has full column rank at every bond", ok_rank)

    # ... and both fail as soon as a bond dimension is not attained.  Take
    # d = (2,2,2) with declared bonds r = (3,2).  Sites 1 and 2 can still be made
    # right-isometric, so the state is a legitimate mixed-canonical MPS with OC at
    # site 0, but Mat_R(C^[0]) is 2 x 3: the thin QR of step K2 does not exist and
    # Psi lies in M_{<=r} and not in M_r.
    def right_iso(rl, rr, d):
        Q, _ = qr(random_tensor(rng, (rr * d, rl)), mode="reduced")
        return ten_L(Q.conj().T, rr, d)

    A0 = random_tensor(rng, (1, 3, 2))
    padded = [A0, right_iso(3, 2, 2), right_iso(2, 1, 2)]
    rows = padded[0].shape[0] * padded[0].shape[2]
    cols = padded[0].shape[1]
    iso_ok = norm(mat_L(padded[1]) @ mat_L(padded[1]).conj().T - np.eye(3)) < 1e-10
    report("T11c a non-attained bond makes the K2 thin QR ill-posed",
           iso_ok and rows < cols,
           f"d = (2,2,2), r = (3,2): sites 1,2 right-isometric, but Mat_R(C^[0]) "
           f"is {rows} x {cols} -- more columns than rows")


def main():
    rng = np.random.default_rng(20261009)
    print("=" * 86)
    t1_iota_isometry(rng)
    t2_state_preservation(rng)
    t3_exactness(rng)
    t3c_theta_identity(rng)
    t3d_proof_steps(rng)
    t4_degenerate(rng)
    t5_two_site_residual(rng)
    t6_N2(rng)
    t7_no_tangent(rng)
    t8_conjecture(rng)
    t9_error_floor(rng)
    t10_truncation(rng)
    t11_qr_wellposed(rng)
    print("=" * 86)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - nfail} / {len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
