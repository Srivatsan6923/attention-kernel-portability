"""Tests for the derived tables whose errors would still look plausible.

Scaling exponents, portability ratios and warm/cold pairs produce numbers even
when rows are paired wrongly, so these tests check the pairing. They run on CPU
and need no results directory.
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
    # T = N^2 exactly, up to 1024. The truncation should show in n_max and
    # leave alpha unchanged.
    d = cells_frame([("g", "fast", cell(n), float(n) ** 2)
                     for n in (256, 512, 1024, 2048)]
                    + [("g", "oomy", cell(n), float(n) ** 2)
                       for n in (256, 512, 1024)])
    s = analysis.scaling(d).set_index("implementation")
    assert np.allclose(s.alpha, 2.0)
    assert s.n_max["oomy"] == 1024 and s.n_max["fast"] == 2048
    # Two points always fit a line exactly, so no exponent is reported.
    assert analysis.scaling(cells_frame(
        [("g", "a", cell(n), float(n)) for n in (256, 512)])).empty


def test_portability_scores_the_winner_at_one_and_drops_unshared_cells():
    d = cells_frame([("g1", "a", cell(512), 10.0), ("g1", "b", cell(512), 20.0),
                     ("g2", "a", cell(512), 40.0), ("g2", "b", cell(512), 10.0),
                     # g1 only. A missing impl must not count as portable.
                     ("g1", "a", cell(512, b=8), 1.0)])
    p = analysis.portability(d).set_index(["gpu_name", "implementation"])
    assert set(p.cell) == {cell(512)}
    assert p.p_ratio[("g1", "a")] == 1.0 and p.p_ratio[("g2", "b")] == 1.0
    assert p.p_ratio[("g1", "b")] == 0.5 and p.p_ratio[("g2", "a")] == 0.25


def test_a_lost_launch_does_not_stop_the_inversion_scan():
    """A backend that lost a launch to an OOM must not stop the scan.

    P0-naive lost its fifth launch in eight RTX 5090 cells. The pair is compared
    on the launches both backends share, so the sign test, the 10% test and the
    interval all use the same launches.
    """
    d = cells_frame([("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 200.0),
                     ("g2", "a", cell(256), 200.0), ("g2", "b", cell(256), 100.0)])
    d["repeat_ids"] = [[0, 1, 2]] * len(d)
    # g1's "a" lost launch 2 and its other launches are much slower. Without
    # pairing the scan would rank on 100.0 and build the interval on 400.0.
    i = d.index[(d.gpu_name == "g1") & (d.implementation == "a")][0]
    d.at[i, "samples"], d.at[i, "repeat_ids"] = [400.0, 400.0], [0, 1]

    out, examined = analysis.inversions(d, "g1", "g2")
    assert examined == 1
    # a is slower on both g1 and g2, so there is no flip.
    assert len(out) == 0
    assert out.attrs["pairs_unpairable"] == 0


def test_shared_launches_still_refuses_a_silent_mismatch():
    """The ledger keeps the strict launch-set check."""
    import pytest
    a, b = np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="launch sets differ"):
        analysis.shared_launches(a, b, [0, 1], [0, 1, 2])
    x, y = analysis.shared_launches(a, b, [0, 1], [0, 1, 2], strict=False)
    assert len(x) == len(y) == 2


def rows_frame(rows):
    d = pd.DataFrame(rows, columns=["cache", "inner_k", "median_us"])
    d["status"], d["gpu_name"], d["implementation"] = "OK", "g", "a"
    d["cell"] = d.cache.map(lambda c: cell(512, c))
    return d


def test_cache_sensitivity_never_pairs_across_inner_k():
    # Only the first of k calls is cold, so a k=3 cold block is mostly warm.
    # Pairing it with a k=1 warm row would show a false speedup.
    d = rows_frame([("warm", 1.0, 100.0), ("cold", 1.0, 150.0),
                    ("cold", 3.0, 60.0)])
    out = analysis.cache_sensitivity(d).set_index("inner_k")
    assert out.sensitivity[1.0] == 0.5
    # Kept but left unpaired.
    assert 3.0 in out.index and pd.isna(out.sensitivity[3.0])
    assert (out.n_pairs == 1).all()


if __name__ == "__main__":
    test_scaling_recovers_the_exponent_and_reports_its_range()
    test_portability_scores_the_winner_at_one_and_drops_unshared_cells()
    test_cache_sensitivity_never_pairs_across_inner_k()
    print("ok")


def test_inversion_denominator_counts_every_pair_compared():
    """The inversion rate divides by every pair compared.

    Two cells with one inversion must give 50%. The non-inverting cell sorts
    second so it would be missed if only inverting rows were counted.
    """
    rows = []
    # cell(256): a faster on g1, b faster on g2, so it inverts by more than 10%.
    rows += [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 200.0),
             ("g2", "a", cell(256), 200.0), ("g2", "b", cell(256), 100.0)]
    # cell(512): a faster on both, so no inversion. Sorts after cell(256).
    rows += [("g1", "a", cell(512), 100.0), ("g1", "b", cell(512), 200.0),
             ("g2", "a", cell(512), 100.0), ("g2", "b", cell(512), 200.0)]
    d, examined = analysis.inversions(cells_frame(rows), "g1", "g2")

    assert examined == 2, f"both pairs were compared, got {examined}"
    assert len(d) == 1 and bool(d.practical.iloc[0])
    assert float(d.practical.sum() / examined) == 0.5
    assert int(d.pairs_examined.iloc[0]) == 2, "row must carry the final total"


def test_inversion_denominator_survives_zero_inversions():
    """With no inversions the frame is empty but the denominator is still set."""
    rows = [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 200.0),
            ("g2", "a", cell(256), 100.0), ("g2", "b", cell(256), 200.0)]
    d, examined = analysis.inversions(cells_frame(rows), "g1", "g2")
    assert len(d) == 0
    assert examined == 1
    assert d.attrs["pairs_examined"] == 1


def test_sig_and_practical_are_intersected_explicitly():
    """A pair must meet both conditions to count."""
    rows = [("g1", "a", cell(256), 100.0), ("g1", "b", cell(256), 104.0),
            ("g2", "a", cell(256), 104.0), ("g2", "b", cell(256), 100.0)]
    d, _ = analysis.inversions(cells_frame(rows), "g1", "g2")
    assert len(d) == 1
    r = d.iloc[0]
    # 4% apart. It inverts but stays under the 10% margin.
    assert not bool(r.practical)
    assert bool(r.sig_and_practical) == (bool(r.sig) and bool(r.practical))


def test_boot_ratio_uses_the_median_and_pairs_process_launches():
    """The bootstrap resamples the median, the same statistic that is reported.

    One slow launch moves a mean much more than a median, so this test would
    fail if the interval were built on the mean.
    """
    a = [100.0, 100.0, 100.0, 100.0, 400.0]   # one slow launch
    b = [100.0] * 5
    lo, hi = analysis._boot_ratio(a, b, ids_a=[0, 1, 2, 3, 4],
                                 ids_b=[0, 1, 2, 3, 4])
    # Both medians are 100, so the ratio is 1 despite the outlier. A mean-based
    # interval would be near 1.6 and exclude 1.
    assert lo <= 1.0 <= hi, f"median-based interval should cover 1, got [{lo}, {hi}]"

    # Identical paired launches give a ratio of exactly 1 on every draw.
    lo2, hi2 = analysis._boot_ratio(b, b, ids_a=[0, 1, 2, 3, 4],
                                   ids_b=[0, 1, 2, 3, 4])
    assert lo2 == hi2 == 1.0


# --- estimator population, n=1, gate scope -----------------------------------

def test_ratio_and_interval_use_one_population():
    """The ratio and its interval come from the same shared launches."""
    a = [100.0, 100.0, 100.0, 5000.0, 5000.0]   # ids 0,1,2,7,8
    b = [100.0] * 5                              # ids 0..4
    ratio, lo, hi, n = analysis.paired_ratio_ci(
        a, b, ids_a=[0, 1, 2, 7, 8], ids_b=[0, 1, 2, 3, 4],
        strict_pairing=False)
    assert n == 3, "only the shared launches are comparable"
    assert ratio == 1.0, "estimate must come from the same 3 launches"
    assert lo == hi == 1.0


def test_mismatched_launch_sets_are_refused_by_default():
    """Different launch sets raise an error by default.

    Compared backends share a launch set in all 2,082 cells here. A sweep that
    breaks this should fail.
    """
    import pytest as _pytest
    with _pytest.raises(ValueError, match="launch sets differ"):
        analysis.paired_ratio_ci([1.0] * 5, [1.0] * 5,
                                 ids_a=[0, 1, 2, 7, 8], ids_b=[0, 1, 2, 3, 4])


def test_one_launch_cannot_produce_a_separated_winner():
    """With one launch the interval is undefined, so it cannot separate.

    Resampling one value gives a zero-width interval that would exclude 1 for
    any ratio.
    """
    ratio, lo, hi, n = analysis.paired_ratio_ci([50.0], [1.0], ids_a=[0], ids_b=[0])
    assert n == 1 and ratio == 50.0
    assert np.isnan(lo) and np.isnan(hi)
    assert analysis.separated(ratio, lo) is False


def test_separation_needs_the_lower_bound_above_one():
    """Separation needs the ratio past the margin and the lower bound above 1."""
    assert analysis.separated(1.5, 1.2) is True
    assert analysis.separated(1.5, 0.9) is False     # interval spans 1
    assert analysis.separated(1.02, 1.01) is False   # under the margin
    assert analysis.separated(1.5, float("nan")) is False


def test_duplicate_launch_ids_are_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="duplicate launch ids"):
        analysis.paired_ratio_ci([1.0, 1.0], [1.0, 1.0], ids_a=[0, 0], ids_b=[0, 1])


def test_gate_verdict_does_not_cross_devices():
    """A correctness pass on one GPU does not verify the class on another.

    gate_class does not include the GPU, so the verdict is propagated per GPU.
    """
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
    """The winner grid ranks medians and not the fastest single launch.

    A backend at [1, 100, 100] us beats one at [10, 10, 10] on the minimum and
    loses on the median.
    """
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
