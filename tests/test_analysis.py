"""Checks for the derived tables that would still look plausible if wrong.

A scaling exponent, a portability ratio and a warm/cold pair all come out as
numbers whatever the pairing does. These assert the pairing, not the arithmetic.
Runs on CPU: no CUDA, no results directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from akp import analysis


def cell(n, cache="warm", b=4):
    return f"prefill|B{b}|Hq32|Hkv32|D128|N{n}|bf16|fwd|eager|{cache}|c1"


def cells_frame(rows):
    d = pd.DataFrame(rows, columns=["gpu_name", "implementation", "cell",
                                    "median_us"])
    d["samples"] = [[u] * 3 for u in d.median_us]
    return d


def test_scaling_recovers_the_exponent_and_reports_its_range():
    # T = N^2 exactly, but only up to 1024: the truncation the score-matrix
    # impls suffer must show up in n_max, not get absorbed into alpha.
    d = cells_frame([("g", "fast", cell(n), float(n) ** 2)
                     for n in (256, 512, 1024, 2048)]
                    + [("g", "oomy", cell(n), float(n) ** 2)
                       for n in (256, 512, 1024)])
    s = analysis.scaling(d).set_index("implementation")
    assert np.allclose(s.alpha, 2.0)
    assert s.n_max["oomy"] == 1024 and s.n_max["fast"] == 2048
    # Two points fit a line exactly; that is not a measured exponent.
    assert analysis.scaling(cells_frame(
        [("g", "a", cell(n), float(n)) for n in (256, 512)])).empty


def test_portability_scores_the_winner_at_one_and_drops_unshared_cells():
    d = cells_frame([("g1", "a", cell(512), 10.0), ("g1", "b", cell(512), 20.0),
                     ("g2", "a", cell(512), 40.0), ("g2", "b", cell(512), 10.0),
                     # g1 only: an impl must not look portable by being absent.
                     ("g1", "a", cell(512, b=8), 1.0)])
    p = analysis.portability(d).set_index(["gpu_name", "implementation"])
    assert set(p.cell) == {cell(512)}
    assert p.p_ratio[("g1", "a")] == 1.0 and p.p_ratio[("g2", "b")] == 1.0
    assert p.p_ratio[("g1", "b")] == 0.5 and p.p_ratio[("g2", "a")] == 0.25


def rows_frame(rows):
    d = pd.DataFrame(rows, columns=["cache", "inner_k", "median_us"])
    d["status"], d["gpu_name"], d["implementation"] = "OK", "g", "a"
    d["cell"] = d.cache.map(lambda c: cell(512, c))
    return d


def test_cache_sensitivity_never_pairs_across_inner_k():
    # Only the first of k calls is truly cold, so a k=3 cold block is mostly
    # warm. Pairing it against a k=1 warm row invents a speedup.
    d = rows_frame([("warm", 1.0, 100.0), ("cold", 1.0, 150.0),
                    ("cold", 3.0, 60.0)])
    out = analysis.cache_sensitivity(d).set_index("inner_k")
    assert out.sensitivity[1.0] == 0.5
    # Kept, not dropped, but unpaired.
    assert 3.0 in out.index and pd.isna(out.sensitivity[3.0])
    assert (out.n_pairs == 1).all()


if __name__ == "__main__":
    test_scaling_recovers_the_exponent_and_reports_its_range()
    test_portability_scores_the_winner_at_one_and_drops_unshared_cells()
    test_cache_sensitivity_never_pairs_across_inner_k()
    print("ok")
