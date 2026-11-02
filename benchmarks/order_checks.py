"""Numerical checks of the step-size expansion behind Proposition 3.

    python benchmarks/order_checks.py

Every claim the proof makes is checked here against the measured step, and nothing goes
into the note that does not pass.  The claims, in the order the proof uses them:

  (1) The pre-truncation step is exactly  Y^1 = exp(h Pi F Pi) Y^0,  with Pi the
      projector onto the kept space V = W_{L-1} tensor C^d.  Checked to roundoff.
  (2) Pi Y^0 = Y^0, which is what makes the h^0 term of e(h)/h equal to -(1-Pi)G.
  (3) The K-step contributes the single tilted direction  M X + (h/2) M^2 X + O(h^2):
      the span of {X, exp(hM)X} is within O(h^2) of the span of {X, M X + (h/2) M^2 X}.
  (4) The resulting orders.  With p extra Krylov directions in the bond basis the
      one-step error is  h delta_0 + O(h^(3+p)),  so for delta_0 = 0 the measured slope
      of e(h)/h is 2 + p.  Checked for p = 0, 1, 2, and for the un-tilted variant
      (K-step direction M X in place of exp(hM) X), which must lose one order.
  (5) The rigorous but loose bound  ||e(h)|| <= h delta_0 + h^2 ||F||^2.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.linalg import norm, qr, svd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bug_mps_checks import (  # noqa: E402
    ten_R, mat_R, iota_matrix, orth_basis, expm_apply, backward_svd_sweep,
    mps_to_vector,
)
from defect_experiment import tfim, tt_svd, defect, N, G_FIELD, T_PREP  # noqa: E402
from scipy.linalg import expm  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def bug_step(mps, F, dt, extra=0, tilt=True, tol=1e-14, probe=None):
    """BUG-MPS sweep; returns (state, pre-truncation state, iota of the final S-step).

    tilt=False replaces the exact K-step solution exp(dt M)X by the plain Krylov
    direction M X, which drops the (dt/2) M^2 X tilt.

    probe=k also returns, for bond k, the gap between span{Q, exp(dt M)X} and
    span{Q, M X + (dt/2) M^2 X}, together with the local dimensions, so that a caller
    can check the comparison is not vacuous (it is vacuous once rl*d <= 2*rr, where
    both spans are the whole local space).
    """
    n = len(mps)
    cur = [A.copy() for A in mps]
    probe_out = None
    for k in range(n - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        Meff = Ik.conj().T @ F @ Ik
        if tilt:
            first = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)
        else:
            first = (Meff @ XC.reshape(-1)).reshape(rl * d, rr)
        cols, v = [first], first.reshape(-1)
        for _ in range(extra):
            v = Meff @ v
            cols.append(v.reshape(rl * d, rr))
        Q, U = qr(XC, mode="reduced")
        Qr = orth_basis(np.hstack([Q] + cols), tol=tol)
        if probe == k:
            MX = (Meff @ XC.reshape(-1)).reshape(rl * d, rr)
            M2X = (Meff @ Meff @ XC.reshape(-1)).reshape(rl * d, rr)
            B = orth_basis(np.hstack([Q, MX + (dt / 2) * M2X]), tol=tol)
            gap = norm(Qr @ Qr.conj().T - B @ B.conj().T, 2)
            probe_out = {"gap": float(gap), "rl_d": rl * d, "rr": rr,
                         "rhat": Qr.shape[1]}
        T = Qr.conj().T @ Q
        cur[k] = ten_R(Qr, rl, d)
        cur[k + 1] = np.einsum("ab,bcp->acp", T,
                               np.einsum("ab,bcp->acp", U, cur[k + 1]))
    k = n - 1
    rl, rr, d = cur[k].shape
    Ik = iota_matrix(cur, k)
    Meff = Ik.conj().T @ F @ Ik
    cur[k] = ten_R(expm_apply(dt * Meff, mat_R(cur[k]).reshape(-1)).reshape(rl * d, rr),
                   rl, d)
    pre = [A.copy() for A in cur]
    out, _ = backward_svd_sweep(cur, 0.0, None)
    return (out, pre, Ik, probe_out) if probe is not None else (out, pre, Ik)


def main():
    F = -1j * tfim(N, G_FIELD)
    up = np.zeros(2 ** N, dtype=complex)
    up[0] = 1.0
    quench = expm(T_PREP * F) @ up
    ds = [2] * N
    Id = np.eye(2 ** N)

    print(f"TFIM N={N}, g={G_FIELD}, t_prep={T_PREP};  ||F|| = {norm(F, 2):.3f}")

    # ---------------------------------------------------------------- (1), (2)
    print("\nthe step is a projected exponential")
    for chi in (2, 4, 8):
        mps = tt_svd(quench, ds, chi)[0] if isinstance(tt_svd(quench, ds, chi), tuple) \
            else tt_svd(quench, ds, chi)
        psi0 = mps_to_vector(mps)
        h = 1e-2
        out, pre, Ik = bug_step(mps, F, h)
        Pi = Ik @ Ik.conj().T
        lhs = mps_to_vector(pre)
        rhs = expm(h * (Pi @ F @ Pi)) @ psi0
        check(f"chi={chi}: Y^1 = exp(h Pi F Pi) Y^0",
              norm(lhs - rhs) < 1e-11 * max(1.0, norm(lhs)),
              f"||diff|| = {norm(lhs - rhs):.2e}")
        check(f"chi={chi}: Pi Y^0 = Y^0",
              norm(Pi @ psi0 - psi0) < 1e-12,
              f"||diff|| = {norm(Pi @ psi0 - psi0):.2e}")

    # ------------------------------------------------------------------- (3)
    # Probed mid-sweep: at the first bonds the augmented columns already fill the local
    # space (rl*d <= 2*rr) and the comparison is vacuous, so we require rl*d > 2*rr.
    print("\nthe K-step supplies one tilted direction, M X + (h/2) M^2 X + O(h^2)")
    # The statement is asymptotic, so we fit the exponent rather than guess a constant:
    # the constant here is large (~5e3), so the two spans are only close for h <~ 1e-3.
    mps, _ = tt_svd(quench, ds, 4)
    hs3 = np.array([1e-3, 3.1623e-4, 1e-4])
    for k in (3, 5):
        gaps, dims = [], None
        for h in hs3:
            _, _, _, pr = bug_step(mps, F, h, probe=k)
            gaps.append(pr["gap"])
            dims = pr
        if dims["rl_d"] <= 2 * dims["rr"]:
            print(f"  [--  ] bond {k}: vacuous "
                  f"(rl*d = {dims['rl_d']} <= 2 rr = {2 * dims['rr']})")
            continue
        p_ = np.polyfit(np.log(hs3), np.log(gaps), 1)[0]
        const = gaps[0] / hs3[0] ** 2
        check(f"bond {k}: subspace gap is O(h^2)", abs(p_ - 2.0) < 0.1,
              f"fitted power {p_:.3f}, constant ~{const:.0f}, "
              f"rl*d = {dims['rl_d']}, rr = {dims['rr']}, rhat = {dims['rhat']}")

    # ------------------------------------------------------------------- (4)
    # The slope is only visible where the h-term dominates delta_0, so for each rule we
    # fit on the largest window in which e(h)/h stays at least 50 delta_0.  On the Ising
    # chain at chi = 8 the floor is 7.3e-10 and clips r->4r; the XXZ chain at chi = 8 is
    # genuinely saturated (delta_0 ~ 1e-14) and shows all four.
    from onestep_ladder import model_xxz, tt_svd as tt_svd_dense   # noqa: E402

    print("\nthe augmentation rule sets the order: slope of e(h)/h is 2 + p")
    grid = np.logspace(-3, -1.2, 7)
    cases = [("Ising", F, tt_svd(quench, ds, 8)[0])]
    Fx, vx, dsx = model_xxz()
    qx = expm(T_PREP * Fx) @ vx
    cases.append(("XXZ", Fx, tt_svd_dense(qx / norm(qx), dsx, 8)))
    for name, Fc, mpsc in cases:
        psi0c = mps_to_vector(mpsc)
        _, _, etac, d0c, _ = defect(mpsc, Fc)
        print(f"  {name} at chi=8: eta = {etac:.3e}, delta_0 = {d0c:.3e}")
        for extra, tilt, want, label in ((0, False, 1, "M X only (no tilt)"),
                                         (0, True, 2, "exp(hM)X (standard r->2r)"),
                                         (1, True, 3, "r->3r"),
                                         (2, True, 4, "r->4r")):
            e = np.array([norm(mps_to_vector(bug_step(mpsc, Fc, h, extra=extra,
                                                      tilt=tilt)[0])
                               - expm(h * Fc) @ psi0c) / h for h in grid])
            keep = e > 50 * max(d0c, 1e-13)
            if keep.sum() < 3:
                print(f"  [--  ] {name} {label}: floored by delta_0, slope not visible")
                continue
            sl = np.polyfit(np.log(grid[keep]), np.log(e[keep]), 1)[0]
            check(f"{name} {label}: slope {want}", abs(sl - want) < 0.15,
                  f"measured {sl:.3f} on {keep.sum()} points")

    # ------------------------------------------------------------------- (5)
    print("\nthe rigorous bound  ||e(h)|| <= h delta_0 + h^2 ||F||^2")
    nF = norm(F, 2)
    for chi in (2, 4):
        mps, _ = tt_svd(quench, ds, chi)
        psi0 = mps_to_vector(mps)
        _, _, _, d0, _ = defect(mps, F)
        worst = 0.0
        for h in (1e-3, 1e-2, 1e-1):
            out, _, _ = bug_step(mps, F, h)
            err = norm(mps_to_vector(out) - expm(h * F) @ psi0)
            worst = max(worst, err / (h * d0 + h ** 2 * nF ** 2))
        check(f"chi={chi}: bound holds", worst < 1.0,
              f"worst ratio error/bound = {worst:.2e}  (bound is loose)")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
