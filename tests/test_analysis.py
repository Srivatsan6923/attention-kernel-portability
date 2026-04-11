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


def test_inversion_denominator_counts_every_pair_compared():
    """The denominator is pairs compared, not the counter at the last flip.

    `examined` was written onto a row only when that pair inverted, so the
    column held snapshots taken at inversion times and main() divided by the
    last one. Two cells with one inversion between them reported 100% where
    the truth is 50%. Order matters: the non-inverting cell sorts second, so
    its comparison used to fall off the end of the denominator entirely.
    """
    rows = []
    # cell(256): a faster on g1, b faster on g2 -> inverts, well past 10%.
    rows += [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 200.0),
             ("g2", "a", cell(256), 200.0), ("g2", "b", cell(256), 100.0)]
    # cell(512): a faster on both -> no inversion, and sorts AFTER cell(256).
    rows += [("g1", "a", cell(512), 100.0), ("g1", "b", cell(512), 200.0),
             ("g2", "a", cell(512), 100.0), ("g2", "b", cell(512), 200.0)]
    d, examined = analysis.inversions(cells_frame(rows), "g1", "g2")

    assert examined == 2, f"both pairs were compared, got {examined}"
    assert len(d) == 1 and bool(d.practical.iloc[0])
    assert float(d.practical.sum() / examined) == 0.5
    assert int(d.pairs_examined.iloc[0]) == 2, "row must carry the final total"


def test_inversion_denominator_survives_zero_inversions():
    """With nothing inverting the frame is empty; the denominator is not."""
    rows = [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 200.0),
            ("g2", "a", cell(256), 100.0), ("g2", "b", cell(256), 200.0)]
    d, examined = analysis.inversions(cells_frame(rows), "g1", "g2")
    assert len(d) == 0
    assert examined == 1
    assert d.attrs["pairs_examined"] == 1


def test_sig_and_practical_are_intersected_explicitly():
    """Requiring both conditions must use their intersection, not either one."""
    rows = [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 104.0),
            ("g2", "a", cell(256), 104.0), ("g2", "b", cell(256), 100.0)]
    d, _ = analysis.inversions(cells_frame(rows), "g1", "g2")
    assert len(d) == 1
    r = d.iloc[0]
    # 4% apart: it inverts, but not past the 10% practical margin.
    assert not bool(r.practical)
    assert bool(r.sig_and_practical) == (bool(r.sig) and bool(r.practical))


def test_boot_ratio_uses_the_median_and_pairs_process_launches():
    """The interval must be built on the statistic that is reported.

    per_cell_median ranks the median of per-process medians, so resampling the
    mean put the interval around a different quantity. One large launch drags
    a mean far more than a median, which is what this pins.
    """
    a = [100.0, 100.0, 100.0, 100.0, 400.0]   # one slow launch
    b = [100.0] * 5
    lo, hi = analysis._boot_ratio(a, b, ids_a=[0, 1, 2, 3, 4],
                                 ids_b=[0, 1, 2, 3, 4])
    # Median of a is 100 and of b is 100, so the ratio sits at 1 despite the
    # outlier. A mean-based interval would sit near 1.6 and exclude 1.
    assert lo <= 1.0 <= hi, f"median-based interval should cover 1, got [{lo}, {hi}]"

    # Perfectly paired identical launches: the ratio is exactly 1 every draw.
    lo2, hi2 = analysis._boot_ratio(b, b, ids_a=[0, 1, 2, 3, 4],
                                   ids_b=[0, 1, 2, 3, 4])
    assert lo2 == hi2 == 1.0


# --- third-review regressions: estimator population, n=1, gate scope --------

def test_ratio_and_interval_use_one_population():
    """Ranking on all repeats while resampling only the shared ones described
    two different quantities. Uneven ids once gave a 50x point ratio beside a
    [0.5, 0.5] interval."""
    a = [100.0, 100.0, 100.0, 5000.0, 5000.0]   # ids 0,1,2,7,8
    b = [100.0] * 5                              # ids 0..4
    ratio, lo, hi, n = analysis.paired_ratio_ci(
        a, b, ids_a=[0, 1, 2, 7, 8], ids_b=[0, 1, 2, 3, 4],
        strict_pairing=False)
    assert n == 3, "only the shared launches are comparable"
    assert ratio == 1.0, "estimate must come from the same 3 launches"
    assert lo == hi == 1.0


def test_mismatched_launch_sets_are_refused_by_default():
    """The paper asserts compared backends share a launch set, which holds in
    all 2,082 cells here. A future sweep that breaks it must fail loudly rather
    than rank on one population and build its interval on another."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="launch sets differ"):
        analysis.paired_ratio_ci([1.0] * 5, [1.0] * 5,
                                 ids_a=[0, 1, 2, 7, 8], ids_b=[0, 1, 2, 3, 4])


def test_one_launch_cannot_produce_a_separated_winner():
    """Resampling a single value returns it every draw, so the interval has
    zero width and 'excludes 1' for any ratio at all."""
    ratio, lo, hi, n = analysis.paired_ratio_ci([50.0], [1.0], ids_a=[0], ids_b=[0])
    assert n == 1 and ratio == 50.0
    assert np.isnan(lo) and np.isnan(hi)
    assert analysis.separated(ratio, lo) is False


def test_separation_needs_the_lower_bound_above_one():
    """The ratio is runner-up/fastest and so >= 1. An upper bound below 1
    contradicts the point estimate; it is not evidence of separation."""
    assert analysis.separated(1.5, 1.2) is True
    assert analysis.separated(1.5, 0.9) is False     # interval spans 1
    assert analysis.separated(1.02, 1.01) is False   # under the margin
    assert analysis.separated(1.5, float("nan")) is False


def test_duplicate_launch_ids_are_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="duplicate launch ids"):
        analysis.paired_ratio_ci([1.0, 1.0], [1.0, 1.0], ids_a=[0, 0], ids_b=[0, 1])


def test_gate_verdict_does_not_cross_devices():
    """gate_class carries no GPU, so propagating on it alone let a pass on one
    device mark an untested class on another as verified."""
    d = pd.DataFrame({
        "gpu_name": ["A", "A", "B"],
        "gate_class": ["cls1", "cls1", "cls1"],
        "correctness_pass": [True, None, None],
    })
    st = analysis.gate_status(d)
    assert list(st) == ["pass", "pass", "ungated"], list(st)
    ev = analysis.gate_evidence(d)
    assert list(ev) == ["direct", "inherited", "none"], list(ev)


def test_a_failure_outranks_a_sibling_pass_on_the_same_device():
    d = pd.DataFrame({
        "gpu_name": ["A", "A"],
        "gate_class": ["cls1", "cls1"],
        "correctness_pass": [True, False],
    })
    assert set(analysis.gate_status(d)) == {"fail"}


def test_winner_grid_ranks_medians_not_the_fastest_launch():
    """idxmin over raw rows picks the fastest single process launch, which is a
    different quantity from the median every other number uses. A backend at
    [1, 100, 100] us beats one at [10, 10, 10] on the minimum and loses on the
    median."""
    from akp import webdata
    df = pd.DataFrame({
        "gpu_name": ["G"] * 6,
        "seq_len": [512] * 6,
        "batch": [1] * 6,
        "implementation": ["A", "A", "A", "B", "B", "B"],
        "median_us": [1.0, 100.0, 100.0, 10.0, 10.0, 10.0],
    })
    got = webdata.winner_grid(df)["G"]
    assert len(got) == 1
    assert got[0]["impl"] == "B", "must rank on the repeat median, not the minimum"
    assert got[0]["us"] == 10.0
