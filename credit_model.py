"""Probability-of-default model for credit card clients (UCI Taiwan dataset).

Run:  python credit_model.py
Writes metrics to reports/metrics.json and charts to reports/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
SEED = 42

# OpenML dataset 42477 is the UCI file with generic column names (x1..x23, y).
COLUMNS = (["limit_bal", "sex", "education", "marriage", "age"]
           + [f"pay_{m}" for m in ["sep", "aug", "jul", "jun", "may", "apr"]]
           + [f"bill_{m}" for m in ["sep", "aug", "jul", "jun", "may", "apr"]]
           + [f"paid_{m}" for m in ["sep", "aug", "jul", "jun", "may", "apr"]]
           + ["default"])
PAY = COLUMNS[5:11]
BILL = COLUMNS[11:17]
PAID = COLUMNS[17:23]

# Missing a default costs far more than turning away a good customer.
# 5:1 is an assumption for illustration, not a figure from the data.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0


def load_data() -> pd.DataFrame:
    raw = fetch_openml(data_id=42477, as_frame=True, parser="auto",
                       data_home=str(ROOT / ".cache")).frame
    df = raw.astype(float)
    df.columns = COLUMNS
    df["default"] = df["default"].astype(int)
    # Codes 0, 5 and 6 are undocumented in the source, so they go to "other" (4).
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    limit = out["limit_bal"].clip(lower=1)
    out["utilisation"] = out["bill_sep"] / limit
    out["avg_utilisation"] = out[BILL].mean(axis=1) / limit
    out["months_late"] = (out[PAY] > 0).sum(axis=1)
    out["worst_delay"] = out[PAY].max(axis=1).clip(lower=0)
    # Share of the bill actually repaid each month, capped at 100%.
    ratios = [(out[p] / out[b].where(out[b] > 0)).clip(0, 1)
              for p, b in zip(PAID[:-1], BILL[1:])]
    out["repay_ratio"] = pd.concat(ratios, axis=1).mean(axis=1).fillna(1.0)
    return out


def ks_statistic(y, p) -> float:
    """Largest gap between the cumulative score curves of defaulters and payers."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_cutoff(y, p):
    """PD cutoff that minimises total misclassification cost."""
    grid = np.linspace(0.05, 0.95, 91)
    costs = [COST_FALSE_NEGATIVE * ((p < t) & (y == 1)).sum()
             + COST_FALSE_POSITIVE * ((p >= t) & (y == 0)).sum() for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def to_score(pd_, base=600, base_odds=50, pdo=20):
    """Convert PD to a scorecard: 600 points at 50:1 odds of paying, +20 points doubles the odds."""
    odds = (1 - pd_) / np.clip(pd_, 1e-6, None)
    factor = pdo / np.log(2)
    return base + factor * (np.log(odds) - np.log(base_odds))


def evaluate(name, y, p):
    cutoff, cost = best_cutoff(y, p)
    flagged = p >= cutoff
    return {
        "model": name,
        "auc": round(roc_auc_score(y, p), 4),
        "gini": round(2 * roc_auc_score(y, p) - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "cost_cutoff_pd": round(cutoff, 2),
        "share_flagged": round(float(flagged.mean()), 4),
        "defaults_caught": round(float(flagged[y == 1].mean()), 4),
        "total_cost": cost,
    }


def plot_roc(y, preds):
    plt.figure(figsize=(5, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve, test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGS / "roc.png", dpi=120)
    plt.close()


def plot_calibration(y, preds):
    plt.figure(figsize=(5, 5))
    for name, p in preds.items():
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(mean_p, frac, "o-", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Predicted PD")
    plt.ylabel("Observed default rate")
    plt.title("Calibration by decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "calibration.png", dpi=120)
    plt.close()


def plot_importance(model, X, y):
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5,
                               random_state=SEED)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values().tail(12)
    plt.figure(figsize=(6, 5))
    imp.plot.barh()
    plt.xlabel("Drop in AUC when shuffled")
    plt.title("Permutation importance (gradient boosting)")
    plt.tight_layout()
    plt.savefig(FIGS / "importance.png", dpi=120)
    plt.close()
    return imp.sort_values(ascending=False).round(4).to_dict()


def plot_score_bands(y, score):
    bands = pd.qcut(score, 10, duplicates="drop")
    rate = pd.Series(y).groupby(bands, observed=True).mean()
    rate.index = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in rate.index]
    plt.figure(figsize=(7, 4))
    rate.plot.bar()
    plt.ylabel("Observed default rate")
    plt.xlabel("Score band (logistic scorecard)")
    plt.title("Default rate falls as the score rises")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "score_bands.png", dpi=120)
    plt.close()


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    df = add_features(load_data())
    X, y = df.drop(columns="default"), df["default"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=SEED)

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=True, random_state=SEED)
    logit.fit(X_tr, y_tr)
    gbm.fit(X_tr, y_tr)

    preds = {"Logistic regression": logit.predict_proba(X_te)[:, 1],
             "Gradient boosting": gbm.predict_proba(X_te)[:, 1]}
    results = [evaluate(name, y_te, p) for name, p in preds.items()]

    plot_roc(y_te, preds)
    plot_calibration(y_te, preds)
    plot_score_bands(y_te, to_score(preds["Logistic regression"]))
    importance = plot_importance(gbm, X_te, y_te)

    out = {"test_rows": int(len(y_te)), "test_default_rate": round(float(y_te.mean()), 4),
           "cost_ratio_fn_to_fp": COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE,
           "results": results, "gbm_importance": importance}
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
