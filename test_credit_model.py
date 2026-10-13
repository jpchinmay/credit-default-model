import numpy as np
import pandas as pd

from credit_model import BILL, COLUMNS, PAID, PAY, add_features, best_cutoff, ks_statistic, to_score


def one_client(**overrides):
    row = dict.fromkeys(COLUMNS, 0.0)
    row.update(limit_bal=10000.0, age=30.0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_features():
    late = one_client(pay_sep=2, pay_aug=1, bill_sep=5000, **{b: 1000.0 for b in BILL[1:]},
                      **{p: 500.0 for p in PAID})
    f = add_features(late).iloc[0]
    assert f["months_late"] == 2
    assert f["worst_delay"] == 2
    assert np.isclose(f["utilisation"], 0.5)
    assert np.isclose(f["repay_ratio"], 0.5)
    # A client with no bills should not break the repayment ratio.
    assert add_features(one_client()).iloc[0]["repay_ratio"] == 1.0


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    cutoff, cost = best_cutoff(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert cost == 0 and 0.2 < cutoff <= 0.8
    # 50:1 odds of paying scores the base 600; doubling the odds adds 20 points.
    assert np.isclose(to_score(np.array([1 / 51]))[0], 600)
    assert np.isclose(to_score(np.array([1 / 101]))[0] - to_score(np.array([1 / 51]))[0], 20, atol=0.3)
