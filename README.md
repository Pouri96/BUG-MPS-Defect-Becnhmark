# BUG-MPS benchmarks

Numerics behind the note "First-order incompleteness of the BUG-MPS integrator".
The code holds a dense reference implementation of the BUG integrator on matrix
product states, the two figure experiments of the note, and the checks that test
its propositions numerically. The note itself is distributed separately.

Everything here is self-contained. It needs only numpy, scipy and matplotlib,
reads no external data, and writes only into `figures/`.

## Setup

```
pip install -r requirements.txt
```

Matplotlib is used by the two figure scripts only. The other scripts need numpy and
scipy. Last run with Python 3.14.1, numpy 2.4.6, scipy 1.18.0 and matplotlib 3.11.0
on Windows 11.

## Running

Run any script from any directory. Each one finds its own folder, and the figure
scripts write to `figures/` next to `benchmarks/`.

```
python benchmarks/defect_experiment.py     # Fig. 1
python benchmarks/onestep_ladder.py        # Fig. 2
python benchmarks/order_checks.py
python benchmarks/eta0_crosscheck.py
python benchmarks/robustness_checks.py
python benchmarks/bug_mps_checks.py
```

All scripts are deterministic (seeds fixed). Measured wall time on one laptop core
is about 6 minutes in total.

| Script | Wall time | Output | Used in |
|---|---|---|---|
| `defect_experiment.py` | 38 s | `figures/defect_tfim.{pdf,png,json}` | Fig. 1 of the note |
| `onestep_ladder.py` | 255 s | `figures/onestep_ladder.{pdf,png,json}` | Fig. 2 of the note |
| `order_checks.py` | 41 s | prints 18 checks | every step of the proof of Prop. 3 (order of the sweep) |
| `eta0_crosscheck.py` | 5 s | prints the defect by independent routes | `delta_0` by independent routes, and `delta_0 <= eta` |
| `robustness_checks.py` | 22 s | prints 13 checks | the robustness argument (`eta_0 <= eps`) |
| `bug_mps_checks.py` | 10 s | prints 24 checks | the dense machinery itself (iota maps, projectors, rank bounds) |

The verification scripts print `[PASS]` or `[ok ]` per check and a final count.

## Layout

```
benchmarks/
  bug_mps_checks.py      dense MPS and BUG-MPS primitives, bug_mps_step, expm_apply.
                         Imported by every other script. Also runs as a self-test.
  tdvp_steps.py          one- and two-site TDVP sweeps, for the comparison runs
  defect_experiment.py   Fig. 1
  onestep_ladder.py      Fig. 2
  order_checks.py        verification only
  eta0_crosscheck.py     verification only
  robustness_checks.py   verification only
figures/                 outputs of the two figure scripts, as used in the note
```

## Reproducing the figures

The files in `figures/` are the ones used in the note. Running the figure scripts
overwrites them.

A rerun reproduces the scalar results of Fig. 1 (`eps`, `eta0`, the bond-by-bond
losses) to about 1e-15, and the data of Fig. 2 to about 4e-12. The two heatmaps in
Fig. 1 show `|G|` and the dropped part in the full SVD basis at one cut. The state
has Schmidt rank 2 there, so the other 14 basis vectors are an arbitrary
completion chosen by LAPACK. The block structure and the block norms of the
heatmaps are identical on any machine, while the individual pixels inside the
null-space blocks can differ from the shipped figure.

## A note on labels

The test labels printed by `bug_mps_checks.py` (T1 to T16) follow an early version
of the note and do not match the numbering of the current manuscript.
