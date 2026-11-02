"""The one-step benchmark of the note: the augmentation rule sets the order.

    python benchmarks/onestep_ladder.py

Writes figures/onestep_ladder.{pdf,png,json} and prints every number quoted in the note.

One step from a state of bond dimension chi, with exact local solves and no truncation
(eps = 0), measured as the error per unit time ||Y^1 - A(h)|| / h.  Nothing is capped:
each method grows its bonds as far as its own rule allows.

  rows     Ising (nearest neighbour); XXZ + transverse field; long-range Z_i Z_j/|i-j|^3
  columns  BUG-MPS with the bond basis augmented by p = 0, 1, 2 extra Krylov directions
           (p = 0 is the standard r -> 2r rule), and two-site TDVP

The error separates into two terms that respond to different things,

    ||Y^1 - A(h)|| / h  =  delta_0(chi)  +  C h^(2+p),

a floor delta_0 set by the bond dimension and a slope set by the augmentation rule.

Two numerical points that are easy to get wrong:

  * the grid reaches h = 1e-5, which is as small as these floors can be read: below it
    error/h divides a roundoff-level error by a tiny step and the curves turn back up.
  * tol is the rank-detection threshold of orth_basis.  The new K-step direction has
    singular value O(h), so at the default tol = 1e-10 it is discarded once h is small
    and the curve settles on a spurious floor (7.8e-7 at chi = 6, in place of the true
    delta_0 = 1.0e-8).  Everything here uses tol = 1e-14; 1e-16 reproduces it.
  * bug_mps_checks.bug_mps_step(complete_qr=True) keeps the full rl*d basis, i.e.
    Pi = 1, and is exact by construction.  It is not a BUG-MPS run.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from numpy.linalg import lstsq, norm, qr, svd
from scipy.sparse.linalg import expm_multiply

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bug_mps_checks import (  # noqa: E402
    ten_R, mat_R, iota_matrix, iota2_matrix, orth_basis, expm_apply,
    backward_svd_sweep, mps_to_vector, right_canonicalise,
)
from tdvp_steps import two_site_tensor, tdvp_step  # noqa: E402
from defect_experiment import defect  # noqa: E402  (eta and delta_0 from the geometry)

L, T_PREP, TOL = 8, 0.6, 1e-14
HS = np.logspace(-5, -0.5, 10)
CHIS = (2, 4, 6, 8)
RULES = ((0, "2r"), (1, "3r"), (2, "4r"))
OUT = os.path.join(os.path.dirname(HERE), "figures")


# --------------------------------------------------------------------------- #
# the integrators
# --------------------------------------------------------------------------- #


def tt_svd(v, ds, chi):
    """Compress a vector to an MPS of bond dimension <= chi, orthogonality at site 0."""
    mps, rl, rest = [], 1, v.reshape(1, -1)
    for d in ds[:-1]:
        U, s, Vh = svd(rest.reshape(rl * d, -1), full_matrices=False)
        r = min(chi, int(np.sum(s > 1e-12 * s[0])))
        mps.append(ten_R(U[:, :r], rl, d))
        rest = s[:r, None] * Vh[:r]
        rl = r
    mps.append(ten_R(rest.reshape(rl * ds[-1], 1), rl, ds[-1]))
    mps = right_canonicalise(mps)
    mps[0] = mps[0] / norm(mps_to_vector(mps))
    return mps


def bug_step(mps, F, dt, extra=0, tol=TOL):
    """One BUG-MPS sweep whose bond basis is [Q | exp(dt M)X | M X, ..., M^extra X].

    extra = 0 is the standard rule: one new direction per old one, so rhat_i <= 2 chi_i.
    Returns (state, peak rank solved with before truncation, final max bond).
    """
    n = len(mps)
    cur = [A.copy() for A in mps]
    rhats = []
    for k in range(n - 1):
        rl, rr, d = cur[k].shape
        Ik = iota_matrix(cur, k)
        XC = mat_R(cur[k])
        Meff = np.asarray(Ik.conj().T @ (F @ Ik))
        XK = expm_apply(dt * Meff, XC.reshape(-1)).reshape(rl * d, rr)
        cols, v = [XK], XK.reshape(-1)
        for _ in range(extra):
            v = Meff @ v
            cols.append(v.reshape(rl * d, rr))
        Q, U = qr(XC, mode="reduced")
        Qr = orth_basis(np.hstack([Q] + cols), tol=tol)
        rhats.append(Qr.shape[1])
        T = Qr.conj().T @ Q
        cur[k] = ten_R(Qr, rl, d)
        cur[k + 1] = np.einsum("ab,bcp->acp", T,
                               np.einsum("ab,bcp->acp", U, cur[k + 1]))

    k = n - 1                                            # S-step on the last site
    rl, rr, d = cur[k].shape
    Ik = iota_matrix(cur, k)
    Meff = np.asarray(Ik.conj().T @ (F @ Ik))
    cur[k] = ten_R(expm_apply(dt * Meff, mat_R(cur[k]).reshape(-1)).reshape(rl * d, rr),
                   rl, d)
    out, _ = backward_svd_sweep(cur, 0.0, None)
    return out, max(rhats), max(A.shape[1] for A in out[:-1])


def tdvp2_step(mps, F, dt, tol=TOL):
    """Two-site TDVP, one left-to-right sweep; same return signature as bug_step.

    Its peak rank is that of the evolved two-site tensor before the SVD truncates it.
    """
    n = len(mps)
    cur = [A.copy() for A in mps]
    peaks = []
    for k in range(n - 1):
        rl, _, dk = cur[k].shape
        _, rr, dk1 = cur[k + 1].shape
        I2 = iota2_matrix(cur, k)
        Theta = two_site_tensor(cur[k], cur[k + 1])
        Meff2 = np.asarray(I2.conj().T @ (F @ I2))
        Theta = expm_apply(dt * Meff2, Theta.reshape(-1)).reshape(rl * dk, dk1 * rr)
        peaks.append(min(rl * dk, dk1 * rr))
        U, s, Vh = svd(Theta, full_matrices=False)
        keep = max(int(np.sum(s > tol * s[0])), 1) if s[0] > 0 else 1
        U, s, Vh = U[:, :keep], s[:keep], Vh[:keep]
        cur[k] = ten_R(U, rl, dk)
        cur[k + 1] = (s[:, None] * Vh).reshape(keep, dk1, rr).transpose(0, 2, 1)
        if k < n - 2:                                    # backward one-site step
            rl1, rr1, d1 = cur[k + 1].shape
            Ik = iota_matrix(cur, k + 1)
            Meff = np.asarray(Ik.conj().T @ (F @ Ik))
            XC = expm_apply(-dt * Meff, mat_R(cur[k + 1]).reshape(-1)).reshape(rl1 * d1, rr1)
            cur[k + 1] = ten_R(XC, rl1, d1)
    out = right_canonicalise(cur)
    return out, max(peaks), max(A.shape[1] for A in out[:-1])


# --------------------------------------------------------------------------- #
# the chains
# --------------------------------------------------------------------------- #


def site_op(o, j, d):
    """Dense: at L = 8, d = 2 the whole space is 256-dimensional."""
    return np.kron(np.kron(np.eye(d ** j), o), np.eye(d ** (L - j - 1)))


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def model_ising():
    """H = -sum Z_j Z_{j+1} - g sum X_j, g = 1, from |0...0>."""
    H = (sum(-site_op(Z, j, 2) @ site_op(Z, j + 1, 2) for j in range(L - 1))
         - sum(site_op(X, j, 2) for j in range(L)))
    v = np.zeros(2 ** L, dtype=complex)
    v[0] = 1.0
    return -1j * H, v, [2] * L


def model_xxz():
    """XXZ with Delta = 1/2 and a transverse field 0.4 X, from a Neel state."""
    H = (sum(site_op(X, j, 2) @ site_op(X, j + 1, 2)
             + site_op(Y, j, 2) @ site_op(Y, j + 1, 2)
             + 0.5 * site_op(Z, j, 2) @ site_op(Z, j + 1, 2) for j in range(L - 1))
         + 0.4 * sum(site_op(X, j, 2) for j in range(L)))
    v = np.zeros(2 ** L, dtype=complex)
    v[int("01" * (L // 2), 2)] = 1.0
    return -1j * H, v, [2] * L


def model_longrange(alpha=3.0):
    """H = -sum_{i<j} Z_i Z_j/|i-j|^alpha - sum X_j, from |0...0>."""
    H = sum(-site_op(Z, i, 2) @ site_op(Z, j, 2) / abs(i - j) ** alpha
            for i in range(L) for j in range(i + 1, L))
    H = H - sum(site_op(X, j, 2) for j in range(L))
    v = np.zeros(2 ** L, dtype=complex)
    v[0] = 1.0
    return -1j * H, v, [2] * L


MODELS = [("ising", "Ising (NN)", model_ising),
          ("xxz", "XXZ + field", model_xxz),
          ("lr", r"long-range $1/r^3$", model_longrange)]


# --------------------------------------------------------------------------- #


def slope(ys, hs, lo=1e-3, hi=1e-2):
    i, j = int(np.argmin(abs(hs - lo))), int(np.argmin(abs(hs - hi)))
    return float(np.log10(ys[j] / ys[i]) / np.log10(hs[j] / hs[i]))


def main():
    data = {"hs": HS.tolist(), "chis": list(CHIS), "L": L, "t_prep": T_PREP,
            "tol": TOL, "models": {}}
    for key, label, build in MODELS:
        F, v0, ds = build()
        q = expm_multiply(T_PREP * F, v0)
        q = q / norm(q)
        err = {t: {} for t in ("1T", "2r", "3r", "4r", "2T")}
        peak = {t: {} for t in ("2r", "3r", "4r", "2T")}
        final = {t: {} for t in ("2r", "3r", "4r", "2T")}
        eta, delta0, trunc, cross = {}, {}, {}, {}
        t0 = time.time()
        for c in CHIS:
            m = tt_svd(q, ds, c)
            p = mps_to_vector(m)
            # how well the start state itself is represented at this bond dimension
            trunc[str(c)] = float(norm(q - p * (p.conj() @ q) / (p.conj() @ p)))
            exact = [expm_multiply(h * F, p) for h in HS]
            # eta and delta_0 from the geometry of the first-order sweep, not from a
            # curve: at large chi the plateau of e(h)/h is the h^2 term, not delta_0.
            _, _, eta_c, d0_c, _ = defect(m, F)
            eta[str(c)], delta0[str(c)] = float(eta_c), float(d0_c)
            for extra, tag in RULES:
                e, pk, fb = [], [], []
                for h, E in zip(HS, exact):
                    out, peak_r, fin_r = bug_step(m, F, h, extra=extra)
                    e.append(float(norm(mps_to_vector(out) - E) / h))
                    pk.append(peak_r)
                    fb.append(fin_r)
                err[tag][str(c)], peak[tag][str(c)], final[tag][str(c)] = e, pk, fb
            e, pk, fb = [], [], []
            for h, E in zip(HS, exact):
                out, peak_r, fin_r = tdvp2_step(m, F, h)
                e.append(float(norm(mps_to_vector(out) - E) / h))
                pk.append(peak_r)
                fb.append(fin_r)
            err["2T"][str(c)], peak["2T"][str(c)], final["2T"][str(c)] = e, pk, fb
            # one-site TDVP, for reference: fixed rank, so no peak or final to record
            err["1T"][str(c)] = [float(norm(mps_to_vector(tdvp_step(m, F, h)) - E) / h)
                                 for h, E in zip(HS, exact)]
            # where the standard rule stops beating one-site TDVP, measured directly
            cross[str(c)] = None
            for h in np.logspace(-5, -1, 120):
                ex1 = expm_multiply(h * F, p)
                eb = norm(mps_to_vector(bug_step(m, F, h)[0]) - ex1) / h
                e1 = norm(mps_to_vector(tdvp_step(m, F, h)) - ex1) / h
                if eb > e1:
                    cross[str(c)] = float(h)
                    break
        data["models"][key] = {"label": label, "err": err, "peak": peak,
                               "final": final, "eta": eta, "delta0": delta0,
                               "trunc": trunc, "crossing": cross}
        print(f"{key}: {time.time() - t0:.0f}s", flush=True)

    report(data)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "onestep_ladder.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    plot(data)


def report(data):
    hs = np.array(data["hs"])
    j = int(np.argmin(abs(hs - 1e-2)))
    print("\nOne step from the t = %.1f state, L = %d, uncapped, tol = %.0e"
          % (data["t_prep"], data["L"], data["tol"]))
    i3 = int(np.argmin(abs(hs - 1e-3)))
    print("\n%-10s %3s | %-10s %-10s | %5s %5s %5s | %-10s | peak rank 2r/3r/4r/2T"
          % ("model", "chi", "eta", "delta_0", "2r", "3r", "4r", "2T at 1e-3"))
    for key, m in data["models"].items():
        E, P = m["err"], m["peak"]
        for c in data["chis"]:
            s = str(c)
            sl = [slope(np.array(E[t][s]), hs) for t in ("2r", "3r", "4r")]
            print("%-10s %3d | %.3e  %.3e | %5.2f %5.2f %5.2f | %.3e | %2d %2d %2d %2d"
                  % (key, c, m["eta"][s], m["delta0"][s], sl[0], sl[1], sl[2],
                     E["2T"][s][i3], max(P["2r"][s]), max(P["3r"][s]),
                     max(P["4r"][s]), max(P["2T"][s])))

    # C is read off the chi = 8 curve, where the h^2 term is exposed and delta_0 is
    # negligible; it is the same constant for every chi, which is the point.
    print("\ncrossover  h* = sqrt((eta - delta_0)/C),  C from the chi = 8 curve at h = 1e-3")
    print("%-10s %-8s | %3s | %-10s %-10s | %-10s %-10s | %-10s"
          % ("model", "C", "chi", "eta", "delta_0", "h* pred", "observed", "trunc err"))
    for key, m in data["models"].items():
        h0 = hs[i3]
        C = float(m["err"]["2r"]["8"][i3]) / h0 ** 2
        data["models"][key]["C"] = C
        for c in data["chis"]:
            s = str(c)
            e_, d0 = m["eta"][s], m["delta0"][s]
            hstar = float(np.sqrt(max(e_ - d0, 0.0) / C))
            obs = m["crossing"][s]
            print("%-10s %8.3f | %3d | %.3e  %.3e | %.3e  %-10s | %.3e"
                  % (key, C, c, e_, d0, hstar,
                     "none" if obs is None else "%.3e" % obs, m["trunc"][s]))


def plot(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.array(data["hs"])
    COL = {2: "#7e57c2", 4: "#2a78d6", 6: "#1baf7a", 8: "#111111"}
    GRID, INK = "#e4e3df", "#0b0b0b"
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "pdf.fonttype": 42, "mathtext.fontset": "cm",
    })
    cols = [("2r", r"BUG-MPS, $r\to2r$ (standard)"), ("3r", r"BUG-MPS, $r\to3r$"),
            ("4r", r"BUG-MPS, $r\to4r$"), ("2T", "two-site TDVP")]
    fig, ax = plt.subplots(3, 4, figsize=(7.0, 5.2), sharex=True, sharey=True)
    handles = None
    for r, (key, label, _) in enumerate(MODELS):
        E = data["models"][key]["err"]
        ref = np.array(E["2r"]["8"])
        for ci, (tag, title) in enumerate(cols):
            a = ax[r, ci]
            a.fill_between(hs, 1e-16, ref, color="#f3c6c6", alpha=0.5, lw=0, zorder=0)
            a.loglog(hs, ref, color="#d14b4b", lw=0.7, ls="--", zorder=1)
            lns = []
            for c in data["chis"]:
                ln, = a.loglog(hs, np.maximum(E[tag][str(c)], 1e-16), "-o", ms=2.4,
                               lw=1.0, color=COL[c], zorder=3)
                lns.append(ln)
            handles = handles or lns
            a.grid(True, color=GRID, lw=0.5)
            a.set_ylim(1e-15, 1e2)
            a.set_axisbelow(True)
            if r == 0:
                a.set_title(title, fontsize=7.5, color=INK)
            if r == 2:
                a.set_xlabel(r"step size $h$")
        ax[r, 0].set_ylabel(r"one-step error $/\,h$")
        ax[r, 3].text(0.96, 0.93, label, transform=ax[r, 3].transAxes, fontsize=7.5,
                      ha="right", va="top",
                      bbox=dict(fc="white", ec="#dddddd", alpha=0.92, pad=2))
    ax[0, 0].text(1.4e-4, 2e-13, "forbidden\n" + r"by $r\to2r$", color="#b03a3a",
                  fontsize=6.5)
    fig.legend(handles=handles, labels=[r"$\chi=%d$" % c for c in data["chis"]],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.004),
               columnspacing=2.2, handlelength=1.7)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(os.path.join(OUT, "onestep_ladder.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, "onestep_ladder.png"), dpi=200)


if __name__ == "__main__":
    main()
