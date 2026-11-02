"""
Independent evaluations of the first-order defect eta_0 for the experiment of the note
(transverse-field Ising chain, N = 8, g = 1, bond dimension 2).

    python benchmarks/eta0_crosscheck.py

Each route uses a different formula of the note:

  A  definition (5.9)        || (I - P_Shat) G ||, Shat from the first-order sweep
  B  Theorem 5.7(d)          || D R_0 ||, R_0 = (I - P_T) G from an explicit tangent basis
  C  Theorem 5.7(b)          || D G ||, no tangent projection (Lemma 5.6 says D kills it)
  D  Theorem 5.7(a)          sqrt(sum_k || E_k G ||^2), telescoping over the bonds
  E  Proposition 5.8 (5.13)  sqrt(sum_k two-site residual_k^2)
  F  Proposition 5.8, first  sqrt(sum_k || E_k Q_k (h_k Psi^0) ||^2) using ONLY the bond
     claim                   term h_k = -Z_k Z_{k+1}; the transverse field is dropped
  G  dynamics                one-step error / dt of the actual integrator, extrapolated
                             to dt -> 0 (Richardson), no projector of the theory used
  H  dynamics                eta(dt) of the finite-dt sweep, extrapolated to dt -> 0
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bug_mps_checks import (  # noqa: E402
    mps_to_vector, iota_matrix, orth_basis, proj, bug_mps_step,
)
from robustness_checks import limit_sweep, Bonds, eps_and_normal  # noqa: E402
from defect_experiment import tfim, tt_svd, N, G_FIELD, T_PREP, CHI  # noqa: E402


def op(o, j, n):
    return np.kron(np.kron(np.eye(2 ** j), o), np.eye(2 ** (n - j - 1)))


def richardson(f, dts):
    """Fit f(dt) = a + b dt + c dt^2 on the given step sizes; return a."""
    A = np.vstack([np.ones_like(dts), dts, dts ** 2]).T
    return np.linalg.lstsq(A, f, rcond=None)[0][0]


def main():
    H = tfim(N, G_FIELD)
    F = -1j * H
    up = np.zeros(2 ** N, dtype=complex)
    up[0] = 1.0
    mps, _ = tt_svd(expm(T_PREP * F) @ up, [2] * N, CHI)
    psi0, Tb, g, R0, eps = eps_and_normal(mps, F)

    cur, Qrs = limit_sweep(mps, F)
    B = Bonds(mps, cur, Qrs)
    S = orth_basis(iota_matrix(cur, N - 1))

    Z = np.diag([1.0, -1.0]).astype(complex)
    routes = {}
    routes["A  definition  |(I-P_Shat) G|"] = norm(g - proj(S) @ g)
    routes["B  Thm 5.7(d)  |D R_0|"] = norm(B.D(R0))
    routes["C  Thm 5.7(b)  |D G|"] = norm(B.D(g))
    routes["D  Thm 5.7(a)  sqrt sum |E_k G|^2"] = np.sqrt(
        sum(norm(B.E(g, k)) ** 2 for k in range(N - 1)))
    routes["E  Prop 5.8    sqrt sum two-site^2"] = np.sqrt(
        sum(B.two_site_residual(g, k) ** 2 for k in range(N - 1)))
    bond_only = []
    for k in range(N - 1):
        hk_psi = -1j * (-(op(Z, k, N) @ op(Z, k + 1, N))) @ psi0      # -i h_k Psi^0
        bond_only.append(norm(B.E(B.Q(hk_psi, k), k)))
    routes["F  Prop 5.8    bond terms only"] = np.sqrt(sum(x * x for x in bond_only))

    dts = np.array([4e-4, 2e-4, 1e-4, 5e-5])
    errs, etas = [], []
    for dt in dts:
        out, rec = bug_mps_step(mps, F, dt, theta=0.0)
        errs.append(norm(mps_to_vector(out) - expm(dt * F) @ psi0) / dt)
        etas.append(norm(g - proj(orth_basis(rec["iota_last"])) @ g))
    routes["G  dynamics    err/dt, dt -> 0"] = richardson(np.array(errs), dts)
    routes["H  dynamics    eta(dt), dt -> 0"] = richardson(np.array(etas), dts)

    ref = routes["A  definition  |(I-P_Shat) G|"]
    print(f"TFIM N={N}, g={G_FIELD}, bond dimension {CHI};  eps = {eps:.15f}")
    print(f"{'route':42s} {'eta_0':>20s}   {'|. - A|':>9s}")
    for name, val in routes.items():
        print(f"{name:42s} {val:20.15f}   {abs(val - ref):9.1e}")
    print(f"eta_0 / eps = {ref / eps:.15f}")
    print("bond-resolved (E: two-site residual, F: bond term only):")
    for k in range(N - 1):
        print(f"  k={k}:  {B.two_site_residual(g, k):.15f}   {bond_only[k]:.15f}")
    print("transverse field alone, sum_k |E_k Q_k (field Psi^0)|: "
          f"{sum(norm(B.E(B.Q(-1j * (-G_FIELD) * op(np.array([[0, 1], [1, 0]], complex), j, N) @ psi0, k), k)) for j in range(N) for k in range(N - 1)):.1e}")


if __name__ == "__main__":
    main()
