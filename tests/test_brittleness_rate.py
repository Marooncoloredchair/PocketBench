import pandas as pd

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged


def test_brittleness_rate_basic():
    df = pd.DataFrame(
        {
            "pocket_id": ["p1", "p1", "p2", "p2"],
            "model_name": ["m", "m", "m", "m"],
            "brittle_invariant": [True, True, False, False],
        }
    )
    s = brittleness_rate_from_flagged(df)
    assert s["total_pairs"] == 2
    assert s["brittle_pairs"] == 1
    assert abs(s["brittleness_rate"] - 0.5) < 1e-9
