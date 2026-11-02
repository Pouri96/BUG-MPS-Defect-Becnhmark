"""
The defect experiment of the note: the first-order defect of BUG-MPS on a
transverse-field Ising chain held at a deliberately insufficient bond dimension.

    python benchmarks/defect_experiment.py

Writes figures/defect_tfim.{pdf,png,json} and prints the numbers quoted in the note.
Setting:

  H = - sum_j Z_j Z_{j+1} - g sum_j X_j,  open chain, N = 8, g = 1,  F = -iH;
  Psi^0 = |0...0> evolved under H to time t_prep, compressed to bond dimension r
  by TT-SVD and normalised (r = 2 unless stated), so the bond dimensions are attained.

Panels, both at bond dimension r = CHI on the Ising chain:
  (a) the direction |G> = F|Y^0> at one cut, as a matrix in the old Schmidt basis on
      both sides, completed to full bases, so the four blocks of the decomposition are
      visible and the corner block |N_k> sits in the lower right;
  (b) the part of |G> that bond k throws away: zero outside the corner block, and a
      definite piece of it.

The quadrature sum of those pieces over the bonds is delta_0, which is the h -> 0
plateau of the chi = 2 curve in figures/onestep_ladder.pdf.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from numpy.linalg import norm, svd
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bug_mps_checks import (  # noqa: E402
    ten_R, mps_to_vector, right_canonicalise, iota_matrix, orth_basis, proj,
    bug_mps_step,
)
from robustness_checks import (  # noqa: E402
    limit_sweep, Bonds, eps_and_normal, corner,
)

N, G_FIELD, T_PREP, CHI = 8, 1.0, 0.6, 2
RANKS = (1, 2, 3, 4)
DTS = np.logspace(-6, -0.5, 12)
OUT = os.path.join(os.path.dirname(HERE), "figures")


def tfim(N, g):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)

    def op(o, j):
        return np.kron(np.kron(np.eye(2 ** j), o), np.eye(2 ** (N - j - 1)))

    H = sum(-op(Z, j) @ op(Z, j + 1) for j in range(N - 1))
    return H + sum(-g * op(X, j) for j in range(N))


def tt_svd(v, ds, chi):
    """Compress a vector to an MPS of bond dimension <= chi; returns (mps, ranks)."""
    mps, rl, rest, ranks = [], 1, v.reshape(1, -1), []
    for d in ds[:-1]:
        U, s, Vh = svd(rest.reshape(rl * d, -1), full_matrices=False)
        r = min(chi, int(np.sum(s > 1e-12 * s[0])))
        mps.append(ten_R(U[:, :r], rl, d))
        rest = s[:r, None] * Vh[:r]
        rl = r
        ranks.append(r)
    mps.append(ten_R(rest.reshape(rl * ds[-1], 1), rl, ds[-1]))
    mps = right_canonicalise(mps)
    mps[0] = mps[0] / norm(mps_to_vector(mps))
    return mps, ranks


def defect(mps, F):
    """eps, eta_0 and the first-order sweep's bond operators for one state."""
    psi0, _, g, _, eps = eps_and_normal(mps, F)
    cur, Qrs = limit_sweep(mps, F)
    B = Bonds(mps, cur, Qrs)
    S = orth_basis(iota_matrix(cur, len(mps) - 1))
    return psi0, g, eps, norm(g - proj(S) @ g), B


def main():
    ds = [2] * N
    F = -1j * tfim(N, G_FIELD)
    up = np.zeros(2 ** N, dtype=complex)
    up[0] = 1.0
    quench = expm(T_PREP * F) @ up

    # (a) one-step error / dt against dt, for each bond dimension
    by_rank = []
    for r in RANKS:
        mps, rk = tt_svd(quench, ds, r)
        psi0, _, eps, eta0, _ = defect(mps, F)
        steps = []
        for dt in DTS:
            out, _ = bug_mps_step(mps, F, dt, theta=0.0)
            err = norm(mps_to_vector(out) - expm(dt * F) @ psi0)
            steps.append({"dt": float(dt), "err_over_dt": float(err / dt)})
        by_rank.append({"r": r, "ranks": rk, "eps": float(eps), "eta0": float(eta0),
                        "steps": steps})

    # bond by bond at bond dimension CHI
    mps, ranks = tt_svd(quench, ds, CHI)
    psi0, g, eps, eta0, B = defect(mps, F)
    lost = [B.bond_defect(g, k) for k in range(N - 1)]
    two_site = [B.two_site_residual(g, k) for k in range(N - 1)]
    corners = [corner(psi0, g, ds, k) for k in range(N - 1)]

    # the matrices of one cut in the Schmidt basis: G itself and the part bond k drops
    kstar = 3
    m = 2 ** (kstar + 1)
    Uf, sv, Vhf = svd(psi0.reshape(m, -1), full_matrices=True)
    Wl, Wr = Uf, Vhf.conj().T                      # first r columns: U_k, V_k
    Gm = Wl.conj().T @ g.reshape(m, -1) @ Wr
    Dm = Wl.conj().T @ B.E(B.Q(g, kstar), kstar).reshape(m, -1) @ Wr
    r_k = ranks[kstar]

    data = {"N": N, "g": G_FIELD, "t_prep": T_PREP, "ranks": ranks,
            "norm_F_psi0": float(norm(g)), "eps": float(eps), "eta0": float(eta0),
            "eta0_over_eps": float(eta0 / eps),
            "bond_lost": [float(x) for x in lost],
            "bond_two_site_residual": [float(x) for x in two_site],
            "bond_corner": [float(x) for x in corners],
            "sum_sq_bond_minus_eta0_sq": float(sum(x * x for x in lost) - eta0 ** 2),
            "cut": kstar, "r_cut": int(r_k),
            "G_cut": np.abs(Gm).tolist(), "D_cut": np.abs(Dm).tolist(),
            "by_rank": by_rank}

    print(f"TFIM N={N}, g={G_FIELD}, t_prep={T_PREP}")
    print("  r   eps        eta_0      eta_0/eps   err/dt at smallest dt")
    for row in by_rank:
        print(f"  {row['r']}   {row['eps']:.3e}  {row['eta0']:.3e}  "
              f"{row['eta0'] / row['eps']:.4f}      {row['steps'][0]['err_over_dt']:.3e}"
              f"   bonds {row['ranks']}")
    print(f"bond by bond at r = {CHI}:  ||F Psi^0|| = {norm(g):.4f}")
    print("  bond k      :  " + "  ".join(f"{k:7d}" for k in range(N - 1)))
    print("  |N_k|       :  " + "  ".join(f"{x:7.4f}" for x in corners))
    print("  |D_k N_k|   :  " + "  ".join(f"{x:7.4f}" for x in lost))
    print("  two-site    :  " + "  ".join(f"{x:7.4f}" for x in two_site))
    print(f"  sum_k |D_k N_k|^2 - eta_0^2 = {data['sum_sq_bond_minus_eta0_sq']:.1e}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "defect_tfim.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    plot(data)


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    INK, INK2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "mathtext.fontset": "cm",
    })
    fig = plt.figure(figsize=(3.4, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.055], wspace=0.42)
    ha = fig.add_subplot(gs[0, 0])
    hb = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # |G> at one cut, in the old Schmidt basis on both sides, and what bond i drops
    kc, rc = data["cut"], data["r_cut"]
    Gm, Dm = np.array(data["G_cut"]), np.array(data["D_cut"])
    top = float(Gm.max())
    vmin = top * 1e-5
    nrm = LogNorm(vmin=vmin, vmax=top)
    m, n = Gm.shape
    for ax, M, ttl in ((ha, Gm, r"(a) $|G\rangle$ at bond $i=%d$" % (kc + 1)),
                       (hb, Dm, r"(b) dropped: $D_i|N_i\rangle$")):
        im = ax.imshow(np.maximum(M, vmin * 0.5), cmap="magma", norm=nrm,
                       interpolation="nearest", aspect="equal")
        ax.axhline(rc - 0.5, color="white", lw=1.0)
        ax.axvline(rc - 0.5, color="white", lw=1.0)
        ax.set_xticks([(rc - 1) / 2, (n + rc - 1) / 2])
        ax.set_xticklabels([r"$\bar P_i$", r"$1-\bar P_i$"], fontsize=6.5)
        ax.set_yticks([(rc - 1) / 2, (m + rc - 1) / 2])
        ax.set_yticklabels([r"$P_i$", r"$1-P_i$"], fontsize=6.5)
        ax.tick_params(length=0, pad=1.5)
        ax.set_title(ttl, fontsize=7, loc="left", color=INK, pad=3)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cx, cy = (n + rc) / 2 - 0.5, (m + rc) / 2 - 0.5
    ha.text(cx, cy, r"corner" "\n" r"$|N_i\rangle$", color="white", fontsize=6.5,
            ha="center", va="center")
    hb.text(cx, cy, "a piece of\nthe corner", color="white", fontsize=6.5,
            ha="center", va="center")
    hb.text(cx, rc / 2 - 0.5, "0", color="white", fontsize=6.5, ha="center", va="center")
    hb.text(rc / 2 - 0.5, cy, "0", color="white", fontsize=6.5, ha="center", va="center")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("|amplitude|", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.5)

    fig.subplots_adjust(left=0.115, right=0.855, top=0.88, bottom=0.14)
    fig.savefig(os.path.join(OUT, "defect_tfim.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "defect_tfim.png"), dpi=200)


if __name__ == "__main__":
    main()
